/*******************************************************************************
 * Logic Regression Demo Code
 * Author: Du Ke  (xadke@cn.ibm.com)
 * Date: 2018-11-20
 * The code just for study.
 *******************************************************************************/
package demo.ai.logic_regression;

import java.util.ArrayList;
import java.util.HashMap;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import demo.ai.utils.Line;
import demo.ai.utils.LineChart;
import demo.ai.utils.Matrix;

public class LogicRegressionMonitor {
	public void showResult(LogicRegression LR, double yAxisMax1, double yAxisMax2) throws Exception {
		
		float[][] parameters = LR.getParameters();
		System.out.println("---------------------------------------------------------------------------------");
		System.out.println("步长梯度：\t" + LR.getStep());
		System.out.println("迭代次数：\t" + LR.getIterationTimes());
		System.out.println("测试总数：\t" + LR.testCount);
		System.out.println("成功数量：\t" + LR.testSuccess + "（" + (LR.testSuccess*100/LR.testCount) + "%）");
		System.out.println("参数结果：\t" + Matrix.matrixToString(parameters).toString());
		System.out.println("---------------------------------------------------------------------------------");
		
		//监控输出
		LineChart chart = new LineChart("收敛曲线", yAxisMax1);
		Line line1 = new Line("收敛百分比");
		Line line2 = new Line("收敛值");
		ArrayList<HashMap> moitor = LR.getMoitorTraining();
		for(int i=0;i<moitor.size();i++) {
			HashMap item = moitor.get(i);
			float times = (Integer)item.get("times") * 1.0f;   //(Float)item.get("times");
			int index = (Integer)item.get("times");
			float lost = (Float)item.get("lost");
			float distAvgPercent = (Float)item.get("distAvgPercent");

			if(i>=10) {
				if(distAvgPercent>0.01) line1.addPoint(times,distAvgPercent);
			}
			lost = lost * LR.testCount;
			
			if((i>10 && i<18) || i>moitor.size()-9){
				System.out.println("【" + (index+1) + "】 L(θ) = " + String.format("%.50f",lost) + "\t收敛百分比：" + String.format("%.4f",distAvgPercent) + " %" );
			}
			if(i==22){
				System.out.println("......");
			}
			
		}
		chart.addLine(line1);
		chart.run();
		System.out.println("---------------------------------------------------------------------------------");
		
		float[][] testX = LR.getTestX();
		float[][] testY = LR.getTestY();
		
		// 预测拟合曲线
		LineChart testChart = new LineChart("预测结果图形显示", yAxisMax2);
		Line testLine1 = new Line("实际值");
		Line testLine2_1 = new Line("实际值为1的点");
		Line testLine2_0 = new Line("实际值为0的点");
		ArrayList<HashMap> moitorTesting = LR.getMoitorTesting();
		for(int i=0;i<moitorTesting.size();i++) {
			HashMap item = moitorTesting.get(i);
			float idx = (Integer)item.get("index") * 1.0f;
			float realY = (Float)item.get("realY");
			float hypotheticalY = (Float)item.get("hypotheticalY");
			float x = (Float)item.get("w");
			float result = (Float)item.get("result");
			String status = (String)item.get("status");
			
			testLine1.addPoint(x,realY);
			if(realY==1 ) testLine2_1.addPoint(x,hypotheticalY);
			if(realY==0 ) testLine2_0.addPoint(x,hypotheticalY);
		}
		testChart.addLine(testLine2_1);
		testChart.addLine(testLine2_0);
		testChart.run();
		
		JFreeChart jfreechart = testChart.getJFreeChart();
		XYPlot plot = (XYPlot) jfreechart.getPlot();
		XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
		renderer.setDrawOutlines(true);
		renderer.setSeriesLinesVisible(0, false);
		renderer.setSeriesLinesVisible(1, false);
	}
}
