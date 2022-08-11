package com.nvidia.spark.rapids.jni;

import java.beans.Transient;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Base64;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ContiguousTable;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.DeviceMemoryBufferView;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class CudaIPCTest {

  void createCppProcess(byte[] handle) {
    ProcessBuilder builder = new ProcessBuilder("/home/bobwang/github/learning/cpp/cmake-build-debug/cuda.main", new String(handle));
    try {
      Process p = builder.start();
      BufferedReader br=new BufferedReader(
          new InputStreamReader(
              p.getInputStream()));
      String line;
      while((line=br.readLine())!=null){
        System.out.println(line);
      }
    } catch (IOException e) {
      System.out.println("00000 " + e);
      throw new RuntimeException(e);
    }
  }

  // working
  @Test
  void testFilter() throws IOException, InterruptedException {
    ColumnVector cv = ColumnVector.fromInts(101, 32, 3, 4, 5);
    Table t1 = new Table(cv);
    Boolean[] maskVals = new Boolean[5];
    Arrays.fill(maskVals, true);
    ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
    Table result = t1.filter(mask);
    writeToFile(result.getColumn(0), "/tmp/cuda.ipc");
  }

  @Test
  void testConcatenate() throws IOException, InterruptedException {
    ColumnVector cv = ColumnVector.fromInts(101, 32);
    ColumnVector cv1 = ColumnVector.fromInts(81, 6);
    Table table = new Table(cv, cv1);
    ColumnVector cv2 = ColumnVector.fromInts(13, 4, 589);
    ColumnVector cv3 = ColumnVector.fromInts(12, 13, 14);
    Table table1 = new Table(cv2, cv3);
    Table result = Table.concatenate(table.contiguousSplit()[0].getTable(), table1.contiguousSplit()[0].getTable());
    writeToFile(result.getColumn(0), "/tmp/cuda.ipc");
    writeToFile(result.getColumn(1), "/tmp/cuda.ipc1");
    System.out.println("------------------");
    Thread.sleep(1000000000);
  }


  // not-working
  @Test
  void testContinuseTable() throws IOException, InterruptedException {
    ColumnVector cv = ColumnVector.fromInts(101, 32, 3, 4, 5);
    ColumnVector cv1 = ColumnVector.fromInts(81, 6, 12, 13, 14);
    Table table = new Table(cv, cv1);
    ContiguousTable split[] = table.contiguousSplit();
    byte[] bytes = CudaIPC.getCudaIpcMemHandle(split[0].getBuffer().getAddress());
    ByteBuffer metaBuffer = split[0].getMetadataDirectBuffer().order(ByteOrder.nativeOrder());
//    PackedColumns x = PackedColumns.parseSerializedColumns(metaBuffer);

//    writeToFile(total, "/tmp/cuda.ipc");

    System.out.println("------------------");
//    Thread.sleep(1000000000);
  }

  void writeToFile(ByteBuffer bb, String name) throws IOException {
    File file = new File(name);
    if (file.exists()) {
      file.delete();
    }

    FileOutputStream fileOutputStream = new FileOutputStream(file);
    FileChannel fc = fileOutputStream.getChannel();
    fc.write(bb);
    fileOutputStream.flush();
    fileOutputStream.close();
  }

  void writeToFile(ColumnVector cv, String name) throws IOException {
    long ptr = cv.getDeviceBufferFor(BufferType.DATA).getAddress();
    byte[] handle = CudaIPC.getCudaIpcMemHandle(ptr);

    CudaIPC.checkGpuMemory(ptr, (int)cv.getRowCount());
    File file = new File(name);
    if (file.exists()) {
      file.delete();
    }

    FileOutputStream fileOutputStream = new FileOutputStream(file);
    fileOutputStream.write(handle);
    fileOutputStream.close();
  }

  @Test
  void test() throws InterruptedException, IOException {

    ColumnVector cv;
    Table table;
//    Table table = Table.readParquet(new File("/home/bobwang/tmp/cuda_ipc/person.parquet"));
//    cv = table.getColumn(0);
//    HostColumnVector hcv = cv.copyToHost();
//    for (int i = 0; i < hcv.getRowCount(); i++) {
//      System.out.print(hcv.getInt(i) + " ");
//    }
//    CudaIPC.checkGpuMemory(cv.getDeviceBufferFor(BufferType.DATA).getAddress(), (int)cv.getRowCount());
//    System.out.println();
    ColumnVector cv2 = ColumnVector.fromInts(7, 8);
    cv = ColumnVector.fromInts(11, 32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    Table t1 = new Table(cv);
//    ContiguousTable x[] = t1.contiguousSplit(2, 5, 9);
//    table = Table.concatenate(x[0].getTable(), x[1].getTable());
//    table = x[1].getTable();
    ColumnVector[] xx = cv.slice(2, 5);
//    cv = table.getColumn(0);
    cv = xx[0];
    DeviceMemoryBufferView dmb = (DeviceMemoryBufferView) cv.getDeviceBufferFor(BufferType.DATA);
//    int offset = dmb.getOffset();
//    System.out.println("offset " + offset);

//    cv = ColumnVector.fromInts(11, 32, 3, 4, 5, 6, 9, 10, 20, 38);
    long ptr = cv.getDeviceBufferFor(BufferType.DATA).getAddress();
    byte[] handle1 = CudaIPC.getCudaIpcMemHandle(ptr);
    for (int i = 0; i < handle1.length; i++) {
      System.out.print(handle1[i] + " ");
    }
    System.out.println();
//    ptr = ptr + 4;
    System.out.println("ptr is: " + String.format("0x%08x", ptr));
    CudaIPC.checkGpuMemory(ptr, 5);
    byte[] handle = CudaIPC.getCudaIpcMemHandle(ptr);
    for (int i = 0; i < handle.length; i++) {
      System.out.print(handle[i] + " ");
    }
    System.out.println();

    File file = new File("/tmp/cuda.ipc");
    if (file.exists()) {
      file.delete();
    }

    FileOutputStream fileOutputStream = new FileOutputStream(file);
    fileOutputStream.write(handle);
    fileOutputStream.close();

    File file1 = new File("/tmp/ipc_handle_encoded.txt");
    if (file1.exists()) {
      file1.delete();
    }

    FileOutputStream fileOutputStream1 = new FileOutputStream(file1);
    fileOutputStream1.write("hello".getBytes());
    ptr = cv.getDeviceBufferFor(BufferType.DATA).getAddress();

    FileWriter fileWriter = new FileWriter(file1);
    String cudaIpcInfo = Base64.getEncoder().encodeToString(handle);
    fileWriter.write(cudaIpcInfo);
    fileWriter.flush();
    fileWriter.close();
    System.out.println("ptr is " + ptr);
    Thread.sleep(1000000000);
  }

  @Test
  void testContiguousSplit() {
    ContiguousTable[] splits = null;
    try (Table t1 = new Table.TestBuilder()
        .column(10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .column(50, 52, 54, 56, 58, 60, 62, 64, 66, null)
        .decimal32Column(-3, 10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .decimal64Column(-8, 50L, 52L, 54L, 56L, 58L, 60L, 62L, 64L, 66L, null)
        .build()) {
      splits = t1.contiguousSplit(2, 5, 9);
      assertEquals(4, splits.length);
      assertEquals(2, splits[0].getRowCount());
      assertEquals(2, splits[0].getTable().getRowCount());
      assertEquals(3, splits[1].getRowCount());
      assertEquals(3, splits[1].getTable().getRowCount());
      assertEquals(4, splits[2].getRowCount());
      assertEquals(4, splits[2].getTable().getRowCount());
      assertEquals(1, splits[3].getRowCount());
      assertEquals(1, splits[3].getTable().getRowCount());
    } finally {
      if (splits != null) {
        for (int i = 0; i < splits.length; i++) {
          splits[i].close();
        }
      }
    }
  }
}
