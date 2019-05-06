/*******************************************************************************
 * Logic Regression Demo Code
 * Author: Du Ke  (xadke@cn.ibm.com)
 * Date: 2018-11-20
 * The code just for study.
 *******************************************************************************/
package demo.ai.utils;

import java.util.ArrayList;
import java.util.Arrays;

/*
 * 数学计算类
 */
public class Calculate {

	public Calculate() {
	}
	
	// Sigmoid 函数
	public static float sigmoid(float x){
		return (float) (1 / (1+Math.exp(-x)));
	}
	
	// Sigmoid 函数1
	public static float[][] sigmoid(float[][] x) throws Exception{
		float[][] result = new float[Matrix.rows(x)][Matrix.cols(x)];
		return sigmoid(x, result);
	}
	
	// Sigmoid 函数2
	public static float[][] sigmoid(float[][] x, float[][] result) throws Exception{
		for(int i=0;i<Matrix.rows(x);i++){
			for(int j=0;j<Matrix.cols(x);j++){
				result[i][j] = sigmoid(x[i][j]);
			}
		}
		return result;
	}
}
