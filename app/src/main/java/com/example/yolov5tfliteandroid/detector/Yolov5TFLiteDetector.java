package com.example.yolov5tfliteandroid.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;


public class Yolov5TFLiteDetector {

    private final Size INPNUT_SIZE = new Size(320, 320);
    private final int[] OUTPUT_SIZE = new int[]{1, 6300, 6};
    private String MODEL_FILE;

    private Interpreter tflite;
    Interpreter.Options options = new Interpreter.Options();

    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public void setModelFile() {
        MODEL_FILE = "best-s-fp16.tflite";
    }

    public Size getInputSize() {
        return this.INPNUT_SIZE;
    }

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try {

            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);


        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * 检测步骤
     *
     * @param bitmap
     * @return
     */
    public List<RectF> detect(Bitmap bitmap) {
//        Bitmap bitmap2 = BitmapFactory.decodeResource(Myapp.application.getResources(), R.drawable.p407);
        TensorImage tfliteInput = new TensorImage(DataType.FLOAT32);
        //先缩放成 320 * 320 像素的图片，再进行归一化 每个数组数据 /255,数据分布在 [0,1]
        ImageProcessor processor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();

        tfliteInput.load(bitmap);
        tfliteInput = processor.process(tfliteInput);
        TensorBuffer output = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);
        if (Objects.isNull(tflite)) return null;
        // 这里tflite默认会加一个的纬度 输入:[1,320,320,3] 输出[1,6300,6]
        tflite.run(tfliteInput.getBuffer(), output.getBuffer());
        //处理过后的数据float[]长度为7 分别是 x1, y1, x2, y2, conditional, area, index
        List<float[]> metaFloat = convertTwoDimensionFloat(output);
        if (metaFloat == null || metaFloat.size() == 0) return null;
        List<float[]> boxes = removeBoxPriority(metaFloat);
        return convertRectF(boxes);
    }


    public List<RectF> convertRectF(List<float[]> boxes) {
        List<RectF> rectFs = new ArrayList<>();
        for (float[] box : boxes) {
            RectF rectF = new RectF(box[0] * 3.375f, box[1] * 4.5f, box[2] * 3.375f, box[3] * 4.5f);
            rectFs.add(rectF);
        }
        return rectFs;
    }


    //转换为二维数组
    public List<float[]> convertTwoDimensionFloat(TensorBuffer buffer) {
        //数据行数
        int line = OUTPUT_SIZE[1];
        //数据的col
        int col = OUTPUT_SIZE[2];
        float[] floatArray = buffer.getFloatArray();
        List<float[]> result = new ArrayList<>();
        int pointer = 0;
        float DETECT_THRESHOLD = 0.5f;
        for (int i = 0; i < line; i++) {
            int index = i * col;
            //先确定目标物体的概率
            float conditional = floatArray[index + 4];

            if (conditional > DETECT_THRESHOLD) {

                //废弃最后1列，最后一列是类别，当前只有牙齿类别
                float[] item = new float[col + 1];
                //x y轴进行旋转调换
                float x = floatArray[index] * INPNUT_SIZE.getWidth();
                float y = floatArray[index + 1] * INPNUT_SIZE.getHeight();
                float w = floatArray[index + 2] * INPNUT_SIZE.getWidth();
                float h = floatArray[index + 3] * INPNUT_SIZE.getHeight();

                //x y是中心点坐标，w，h是框的长度和宽度
                float top_x = Math.max(0, x - w / 2.0f);
                float top_y = Math.max(0, y - h / 2.0f);
                float bottom_x = Math.min(INPNUT_SIZE.getWidth(), x + w / 2.0f);
                float bottom_y = Math.min(INPNUT_SIZE.getHeight(), y + h / 2.0f);

                item[0] = top_x;
                item[1] = top_y;
                item[2] = bottom_x;
                item[3] = bottom_y;
                item[4] = conditional;
                item[5] = (bottom_x - top_x + 1) * (bottom_y - top_y + 1);
                //最后一位存放数组的索引
                item[6] = pointer;
                result.add(item);
                pointer++;
            }
        }
        return result;
    }

    public List<float[]> removeBoxPriority(List<float[]> dets) {
        //优先级队列根据概率进行排序，定义队列排序规则
        PriorityQueue<float[]> priorityQueue = new PriorityQueue<>(dets.size(), (a, b) -> -Float.compare(a[4], b[4]));
        priorityQueue.addAll(dets);
        List<float[]> keep = new ArrayList<>();

        float IOU_THRESHOLD = 0.5f;
        while (!priorityQueue.isEmpty()) {
            //取出头部元素
            float[] select = priorityQueue.peek();
            if (select == null) {
                break;
            }


            keep.add(select);
            //遍历队列中所有的元素
            Iterator<float[]> it = priorityQueue.iterator();
            while (it.hasNext()) {
                float[] item = it.next();
                float max_x = Math.max(select[0], item[0]);
                float max_y = Math.max(select[1], item[1]);
                float min_x = Math.min(select[2], item[2]);
                float min_y = Math.min(select[3], item[3]);

                float overlap = Math.max(0, min_x - max_x + 1) * Math.max(0, min_y - max_y + 1);

                Float area = item[5];
                Float selectArea = select[5];
                float iou = overlap / (selectArea + area - overlap);

                if (iou > IOU_THRESHOLD) {
                    it.remove();
                }
            }

        }
        return keep;
    }

    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
            Log.i("tfliteSupport", "using nnapi delegate.");
        }
    }


//    public void addNnapi() {
//        // Initialize interpreter with NNAPI delegate for Android Pie or above
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(false);
//            nnApiOptions.setUseNnapiCpu(true);
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            NnApiDelegate nnApiDelegate = new NnApiDelegate(nnApiOptions);
//            options.addDelegate(nnApiDelegate);
//        }

//}


    /**
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        CompatibilityList compatibilityList = new CompatibilityList();
        if(compatibilityList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i("tfliteSupport", "using gpu delegate.");
        } else {
            addThread(4);
        }
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}
