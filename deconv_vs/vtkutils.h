#pragma once
#include "common.h"
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);
#include <vtkDoubleArray.h>
#include <vtkMath.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkSmartPointer.h>

#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkChartLegend.h>
#include <vtkPlot.h>
#include <vtkTable.h>
#include <vtkPen.h>

#include <vtkTimerLog.h>

/**************************************************************************************************
 * Fn:	int plot1D(std::vector<double> _data, const char* dataName);
 *
 * Plot 1 d.
 *
 * Author:	Lccur
 *
 * Date:	2017/8/6
 *
 * Parameters:
 * _data -    	The data.
 * dataName - 	Name of the data.
 *
 * Returns:	An int.
 */

int plot1D(std::vector<double> _data, const char* dataName);