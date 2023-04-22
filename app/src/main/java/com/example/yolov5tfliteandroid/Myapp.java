package com.example.yolov5tfliteandroid;

import android.app.Application;

public class Myapp extends Application {

    public static Application application;

    @Override
    public void onCreate() {
        super.onCreate();
        application = this;
    }
}
