import datetime

from sklearn import datasets

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

from evidently.report import Report
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import ReportFilter
from evidently.metrics import ColumnDistributionMetric
from evidently.metrics import DatasetCorrelationsMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetSummaryMetric
from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix

import datetime
import pandas as pd
from evidently.report import Report
from evidently.ui.workspace import WorkspaceBase
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import ReportFilter
from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix

import pandas as pd

ref_data = pd.read_csv('/Data/ref_data_test.csv')
prod_data = pd.read_csv('/Data/prod_data.csv')

WORKSPACE = "workspace"
YOUR_PROJECT_NAME = "Marbre anomaly Project"
YOUR_PROJECT_DESCRIPTION = "Projet sur la détection d'anomalie "

# Fonction pour créer un rapport avec des métriques de classification
def create_report(i: int):
    data_report = Report(
        metrics=[
            DatasetSummaryMetric(),
            ColumnDistributionMetric(column_name="target"),
            ColumnDistributionMetric(column_name="prediction"),
            ClassificationQualityMetric(),
            DatasetCorrelationsMetric(),
            ClassificationConfusionMatrix(),
            ColumnSummaryMetric(column_name="target"),
            ColumnSummaryMetric(column_name="prediction"),
            ClassificationClassBalance(),

        ],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )
    data_report.run(reference_data=ref_data, current_data=prod_data)
    return data_report

# Fonction pour créer un projet
def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Nombre de données dans Prod_data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path=DatasetSummaryMetric.fields.current.number_of_rows,
            ),
            text="",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Nombre de données dans Ref_Data_test",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path=DatasetSummaryMetric.fields.reference.number_of_rows,
                legend="Ref Data",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Accuracy Prod Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.current.accuracy,
                legend="Accuracy",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Accuracy Ref Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.reference.accuracy,
                legend="Accuracy",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Precision Prod Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.current.precision,
                legend="Precision",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Precision Ref Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.reference.precision,
                legend="Precision",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Recall Prod Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.current.recall,
                legend="Recall",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Recall Ref Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.reference.recall,
                legend="Recall",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="F1 Prod Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.current.f1,
                legend="F1",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="F1 Ref Data",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path=ClassificationQualityMetric.fields.reference.f1,
                legend="F1",
            ),
            text="",
            agg=CounterAgg.LAST, 
            size=1,
        )
    )

    project.save()
    return project

# Fonction pour créer un projet de démonstration
def create_demo_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    for i in range(0, 5):
        report = create_report(i=i)
        ws.add_report(project.id, report)

if __name__ == "__main__":
    create_demo_project(WORKSPACE)
