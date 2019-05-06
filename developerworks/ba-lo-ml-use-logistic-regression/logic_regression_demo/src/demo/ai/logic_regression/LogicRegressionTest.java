/*******************************************************************************
 * Logic Regression Demo Code
 * Author: Du Ke  (xadke@cn.ibm.com)
 * Date: 2018-11-20
 * The code just for study.
 *******************************************************************************/
package demo.ai.logic_regression;

import java.util.ArrayList;

import demo.ai.utils.MatrixDataFile;

public class LogicRegressionTest {
	
	public static void testStatlogHeart() throws Exception {
		System.out.println("建立心脏病预测逻辑回归模型");
		
		// 数据文件
		MatrixDataFile dataSource = new MatrixDataFile("/data/StatlogHeart.csv"); 
		dataSource.featureScaling();
		
		// 处理数据
		float[][] Y = dataSource.getY();
		for(int i=0;i<Y.length;i++){
			Y[i][0] = (Y[i][0]==1) ? 0:1 ;
		}
		
		// 训练与测试数据集
		ArrayList<float[][]> trainData = dataSource.getFirst(200);
		ArrayList<float[][]> testData  = dataSource.getLast(100);
		
		LogicRegression LR = new LogicRegression();
		LR.setXY(trainData.get(0),trainData.get(1));
		LR.train(new float[]{1,1,1,1,1,  1,1,1,1,1,  1,1,1,  1}, 0.0015f, 2000);
		
		LR.test(LogicRegression.METHOD_GRADIENT_DESCENT,testData.get(0), testData.get(1));

		// 监控输出
		LogicRegressionMonitor monitor = new LogicRegressionMonitor();
		monitor.showResult(LR,100,1.2);
	}
	
	public static void main(String[] args) throws Exception {
		testStatlogHeart();
	}
}
