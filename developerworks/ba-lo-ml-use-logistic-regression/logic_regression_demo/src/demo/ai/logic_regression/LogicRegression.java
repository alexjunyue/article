/*******************************************************************************
 * Logic Regression Demo Code
 * Author: Du Ke  (xadke@cn.ibm.com)
 * Date: 2018-11-20
 * The code just for study.
 *******************************************************************************/
package demo.ai.logic_regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import demo.ai.utils.Calculate;
import demo.ai.utils.Matrix;

public class LogicRegression {
	
	public static final int METHOD_GRADIENT_DESCENT = 0;	// 梯度下降法
	public static final int METHOD_NORMAL_EQUATIONS = 1;	// 正规方程法
	
	private float[][] x;					// 特征矩阵
	private float[][] parameters;			// 参数矩阵
	private float[][] y;					// 标签矩阵(只用第一1列) 0 or 1
	private float step = 0.0001f ;			// 梯度 a
	private int iterationTimes = 10 ;		// 迭代次数

	private int m ;							// 特征矩阵x的行数
	private int n ;							// 特征矩阵x的列数
	private float[][] xT ;					// 特征矩阵x的转置矩阵	= Matrix.transpose(x);
	private float[][] a_times_xT ;			// 梯度 a 与 xT 的乘积 = Matrix.constantProduct(a, xT);
	private float[][] parametersT ;			// 参数矩阵的转置矩阵 = Matrix.transpose(parameters);
	
	private float[][] A ;					// 中间计算矩阵 A = new float[m][1];
	private float[][] g ;					// 中间计算矩阵 g= new float[m][1];
	private float[][] E ;					// 中间计算矩阵 E= new float[m][1];
	private float[][] a_times_xT_times_E ;	// 中间计算矩阵 a*xT*E = new float[n][1];

	private float[][] testX ;				// 测试数据集合x
	private float[][] testY ;				// 测试数据集合y
	
	private int moitorCount = 100 ;												// 监控输出数据的数量
	private ArrayList<HashMap> moitorTraining = new ArrayList<HashMap>();		// 训练监控数据
	private ArrayList<HashMap> moitorTesting ;									// 测试监控数据
	
	public LogicRegression() {
	}
	
	public float[][] getX() {
		return x;
	}

	public void setX(float[][] x) throws Exception {
		this.x = x;
		this.xT = Matrix.transpose(x);
		this.m = x.length;
		this.n = x[0].length;
		
		this.parameters = new float[1][n];
		this.A = new float[m][1];
		this.g = new float[m][1];
		this.E = new float[m][1];
		this.a_times_xT_times_E = new float[n][1];
	}

	public float[][] getParameters() {
		return parameters;
	}

	public void setParameters(float[][] parameters) {
		this.parameters = parameters;
	}

	public float[][] getY() {
		return y;
	}

	public void setY(float[][] y) {
		this.y = y;
	}

	public float getStep() {
		return step;
	}

	public void setStep(float alpha) {
		this.step = alpha;
	}
	
	public int getIterationTimes() {
		return iterationTimes;
	}

	public void setIterationTimes(int iterationTimes) {
		this.iterationTimes = iterationTimes;
	}
	
	public void setXY(float[][] x, float[][] y) throws Exception {
		setX(x);
		setY(y);
	}
	
	public float[][] getTestX() {
		return testX;
	}

	public void setTestX(float[][] testX) {
		this.testX = testX;
	}

	public float[][] getTestY() {
		return testY;
	}

	public void setTestY(float[][] testY) {
		this.testY = testY;
	}
	
	public int getMoitorCount() {
		return moitorCount;
	}

	public void setMoitorCount(int moitorCount) {
		this.moitorCount = moitorCount;
	}

	public ArrayList<HashMap> getMoitorTraining() {
		return moitorTraining;
	}

	public void setMoitorTraining(ArrayList<HashMap> moitorTraining) {
		this.moitorTraining = moitorTraining;
	}
	
	public ArrayList<HashMap> getMoitorTesting() {
		return moitorTesting;
	}

	public void setMoitorTesting(ArrayList<HashMap> moitorTesting) {
		this.moitorTesting = moitorTesting;
	}
	
	public void train(float[] initParameters, float initAlpha, int iterationTimes) throws Exception{
		float[] params;
		float hx;
		float beforeL=0;
		float lost=1;
		float percent = 0 ;
		float p;
		float convergency = 0 ;
		float convergencyBefore = 0 ;
		int moitorStep = (iterationTimes / moitorCount);
		float moitorDiv;
		
		// 初始参数矩阵
		for(int i=0;i<parameters.length;i++){
			if(i<initParameters.length) parameters[0][i] = initParameters[i];
		}
		
		// 设置梯度
		this.setStep(initAlpha);
		
		// 设置迭代次数
		this.setIterationTimes(iterationTimes);
		
		// 转置参数矩阵
		parametersT = Matrix.transpose(parameters);
		
		// 中间矩阵 a * xT
		a_times_xT = Matrix.times(this.step, xT);
		
		// 迭代
		for(int times= 0; times<iterationTimes; times++){
			// (1) 求 A = x * parameters;
			Matrix.times(x, parametersT, A);
		
			// (2) 求 E = h(x) - y
			Calculate.sigmoid(A, g);
			Matrix.add(g, -1, y, E);
			
			// (3) 求 parametersT := parametersT - a * xT * E
			Matrix.times(a_times_xT, E, a_times_xT_times_E);
			Matrix.add(parametersT, -1, a_times_xT_times_E, parametersT);
			
			
			// 监控输出，求下降程度
			lost=1;
			params = Matrix.matrixToColVector(parametersT);
			for(int i=0;i<x.length;i++){
				hx = Calculate.sigmoid(Matrix.dotProduct(x[i], params));
				p = (y[i][0]==0) ? 1 - hx : hx ;
				lost = lost * p;
			}
			if(beforeL>0){
				convergency = lost - beforeL;
				percent = ((lost - beforeL) / beforeL ) * 100 ;
			}
			beforeL = lost;
			convergencyBefore = convergency;
			
			// 监控数据
			moitorDiv = times%moitorStep ;
			if(times<10 || times>iterationTimes-10 || (moitorDiv==0.0)){
				HashMap monitorItem = new HashMap();
				monitorItem.put("times", times);
				monitorItem.put("lost", lost);
				monitorItem.put("distAvgPercent", percent);
				monitorItem.put("parameters", Matrix.copy(parameters));  // 记录本次参数
				moitorTraining.add(monitorItem);
			}
		}
		parameters = Matrix.transpose(parametersT);
	}
	
	// Test parameters
	public int testCount 	= 0;
	public int testSuccess 	= 0;
	public int testFail 	= 0;
	public int testUnknown 	= 0;
	public float threshold = 0.3f;
	
	public void test(int method, float[][] testX, float[][] testY){
		float result;
		String status;
		moitorTesting = new ArrayList<HashMap>();
		testCount = testX.length;
		testSuccess = 0;
		testFail = 0;
		testUnknown = 0;
		this.testX = testX;
		this.testY = testY;
		float[] p = new float[parameters[0].length];
		for(int i=0;i<parameters[0].length;i++) p[i] = parameters[0][i];
		for(int i=0;i<testX.length;i++){
			float[] x = testX[i];
			float y = testY[i][0];
			float w = Matrix.dotProduct(x, p);  // x * p
			float h_y = w ; // 默认正规方程
			if(method==this.METHOD_GRADIENT_DESCENT){ // 梯度下降
				h_y = Calculate.sigmoid(w);
			}else if(method==this.METHOD_NORMAL_EQUATIONS){ // 正规方程
				h_y = w;
			}else{
				h_y = w;
			}
			
			if(h_y<threshold){
				result = 0 ;
			}else if(h_y>threshold){
				result = 1 ;
			}else{
				result = 0.5f ;
			}
			
			if(result==y){
				status = "YES";
				testSuccess++;
			}else{
				if(result == 0.5f){
					status = "---";
					testUnknown++;
				}else{
					status = "NO";
					testFail++;
				}
			}
			
			HashMap monitorItem = new HashMap();
			monitorItem.put("index", i);
			monitorItem.put("realY", y);
			monitorItem.put("w", w);
			monitorItem.put("hypotheticalY", h_y);
			monitorItem.put("result", result);
			monitorItem.put("status", status);
			moitorTesting.add(monitorItem);
		}
	}
}
