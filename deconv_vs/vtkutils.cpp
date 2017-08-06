#include "vtkutils.h"

int plot1D(std::vector<double> _data, const char* dataName)
{
	int dlen = _data.size();
	vtkSmartPointer<vtkTable> table =
		vtkSmartPointer<vtkTable>::New();

	vtkSmartPointer<vtkDoubleArray> arrX =
		vtkSmartPointer<vtkDoubleArray>::New();

	vtkSmartPointer<vtkDoubleArray> arrY =
		vtkSmartPointer<vtkDoubleArray>::New();

	arrX->SetName("X Axis");
	arrY->SetName(dataName);
	table->AddColumn(arrX);
	table->AddColumn(arrY);

	table->SetNumberOfRows(dlen);
	for (int i = 0; i < dlen; i++) {
		std::cout << "idx: " << i << "\t" << _data[i] << std::endl;
		table->SetValue(i, 0, (double)i);
		table->SetValue(i, 1, _data[i]);
	}

	// Setup the view
	vtkSmartPointer<vtkContextView>view =
		vtkSmartPointer<vtkContextView>::New();
	view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

	// Add plots
	vtkSmartPointer<vtkChartXY> chart =
		vtkSmartPointer<vtkChartXY>::New();
	view->GetScene()->AddItem(chart);
	vtkPlot *line = chart->AddPlot(vtkChart::LINE);

	line->SetInputData(table, 0, 1);
	line->SetColor(0, 255, 0, 255);
	line->SetWidth(2.0);

	view->GetInteractor()->Initialize();
	view->GetInteractor()->Start();

	return EXIT_SUCCESS;
}