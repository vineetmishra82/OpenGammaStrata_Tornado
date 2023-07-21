package com.kpts.port23;

import java.util.Random;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

public class Main {
	
    private static final int WARMING_UP_ITERATIONS = 15;

    private static void matrixMultiplication(final float[] matrixA, final float[] matrixB, final float[] result, final int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += matrixA[(i * size) + k] * matrixB[(k * size) + j];
                }
                result[(i * size) + j] = sum;
            }
        }
    }

    public static void main(String[] args) throws NumberFormatException {
    	
    	System.out.println("starting project.....");

        int size = 512;
        if (args.length >= 1) {
            size = Integer.parseInt(args[0]);
        }

        System.out.println("Computing MxM of " + size + "x" + size);
        System.out.println("Inside main method");

        float[] matrixA = new float[size * size];
        float[] matrixB = new float[size * size];
        float[] matrixC = new float[size * size];
        float[] resultSeq = new float[size * size];

        Random r = new Random();
        IntStream.range(0, size * size).parallel().forEach(idx -> {
            matrixA[idx] = r.nextFloat();
            matrixB[idx] = r.nextFloat();
        });

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA, matrixB) //
                .task("t0", Main::matrixMultiplication, matrixA, matrixB, matrixC, size) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC); //

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        //executor.withWarmUp();

        // 1. Warm up Tornado
        for (int i = 0; i < WARMING_UP_ITERATIONS; i++) {
            executor.execute();
        }

        // 2. Run parallel on the GPU with Tornado
        long start = System.currentTimeMillis();
        executor.execute();
        long end = System.currentTimeMillis();

        // Run sequential
        // 1. Warm up sequential
        for (int i = 0; i < WARMING_UP_ITERATIONS; i++) {
            matrixMultiplication(matrixA, matrixB, resultSeq, size);
        }

        // 2. Run the sequential code
        long startSequential = System.currentTimeMillis();
        matrixMultiplication(matrixA, matrixB, resultSeq, size);
        long endSequential = System.currentTimeMillis();

        // Compute GigaFLOPS and performance
        long msecGPUElapsedTime = (end - start);
        long msecCPUElaptedTime = (endSequential - startSequential);
        double flops = 2 * Math.pow(size, 3);
        double gpuGigaFlops = (1.0E-9 * flops) / (msecGPUElapsedTime / 1000.0f);
        double cpuGigaFlops = (1.0E-9 * flops) / (msecCPUElaptedTime / 1000.0f);
        double speedup = (double) (endSequential - startSequential) / (double) (end - start);

        String formatGPUFGlops = String.format("%.2f", gpuGigaFlops);
        String formatCPUFGlops = String.format("%.2f", cpuGigaFlops);

        TornadoDeviceType deviceType = executor.getDevice(0).getDeviceType();

        // @formatter:off
        String buffer = "\tSingle Threaded CPU Execution: " + formatCPUFGlops + " GFlops, Total time = " + (endSequential - startSequential) + " ms" +
                "\n\tTornadoVM Execution on " + deviceType + " (Accelerated): " + formatGPUFGlops + " GFlops, Total Time = " + (end - start) + " ms" +
                "\n\tSpeedup: " + speedup + "x" +
                "\n\tVerification " + verify(matrixC, resultSeq, size) + "\n";
        // @formatter:on

        System.out.println(buffer);
    	
    	
    }

    private static boolean verify(float[] par, float[] seq, int size) {
        boolean check = true;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {

                if (Math.abs(par[i * size + j] - seq[i * size + j]) > 0.1f) {
                    check = false;
                    break;
                }
            }
        }
        return check;
    }

}
