package com.nvidia.spark.rapids.jni;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.ColumnVector;


public class CudaIPCTest {

  @Test
  void test() {

    ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4, 5);
    byte[] handle = CudaIPC.getCudaIpcMemHandle(cv.getNativeView());
    System.out.println(handle);
  }
}
