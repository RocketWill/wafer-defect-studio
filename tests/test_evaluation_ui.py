import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTableWidget

from wafer_defect_studio.evaluation_run import EvaluationDecision, EvaluationRecord
from wafer_defect_studio.main_window import MainWindow


class _DecisionService:
    def __init__(self, decisions):
        self.calls = []
        self._decisions = list(decisions)

    def append_decision(self, evaluation_id, status, actor, *, criteria=None, notes=""):
        self.calls.append((evaluation_id, status, actor, dict(criteria or {}), notes))
        decision = EvaluationDecision(
            f"decision-{len(self._decisions) + 1}",
            evaluation_id,
            status,
            actor,
            "2026-08-22T00:00:00+00:00",
            dict(criteria or {}),
            False,
            notes,
        )
        self._decisions.append(decision)
        return decision


class EvaluationUiTest(unittest.TestCase):
    def test_grid_metrics_targets_history_and_explicit_approval(self):
        app = QApplication.instance() or QApplication([])
        evaluation = EvaluationRecord(
            "evaluation-1",
            "run-1",
            "snapshot-1",
            "split-1",
            "2026-08-22T00:00:00+00:00",
            {"torch": "2.8"},
            {"minimum_recall_target": 0.95},
            {
                "macro_f1": 0.4,
                "per_class": [
                    {
                        "class_name": "scratch",
                        "precision": 0.5,
                        "recall": 0.8,
                        "f1": 0.62,
                        "support": 5,
                        "fpr": 0.1,
                        "true_positive": 4,
                        "true_negative": 9,
                        "false_positive": 1,
                        "false_negative": 1,
                    }
                ],
            },
            {
                "min_recall_target": 0.95,
                "per_class": [
                    {
                        "class_name": "scratch",
                        "policy": "min_recall",
                        "threshold": 0.9,
                        "target": 0.95,
                        "target_satisfied": False,
                    }
                ],
            },
            False,
            "test split",
        )
        candidate = EvaluationDecision(
            "decision-1",
            "evaluation-1",
            "candidate",
            "system",
            "2026-08-22T00:00:00+00:00",
            {"minimum_recall_target": 0.95},
            False,
            "created",
        )
        service = _DecisionService([candidate])
        window = MainWindow()
        window.configure_evaluation(evaluation, (candidate,), decision_service=service)
        window.show()
        app.processEvents()

        metrics = window.findChild(QTableWidget, "evaluationMetricsTable")
        history = window.findChild(QTableWidget, "evaluationDecisionHistoryTable")
        target = window.findChild(QLabel, "evaluationTargetStatusLabel")
        wafer = window.findChild(QLabel, "waferEvaluationPlaceholderLabel")
        self.assertEqual(metrics.rowCount(), 1)
        self.assertEqual(metrics.item(0, 0).text(), "scratch")
        self.assertIn("UNMET", target.text())
        self.assertNotIn("Satisfied", target.text())
        self.assertIn("basic structure", wafer.text())
        self.assertEqual(history.rowCount(), 1)
        self.assertEqual(history.item(0, 2).text(), "Candidate")

        actor = window.findChild(QLineEdit, "evaluationApprovalActorEdit")
        notes = window.findChild(QPlainTextEdit, "evaluationApprovalNotesEdit")
        approve = window.findChild(QPushButton, "approveEvaluationButton")
        self.assertFalse(approve.isEnabled())
        actor.setText("engineer")
        notes.setPlainText("Explicit approval after review")
        approve.click()
        self.assertIn("allow_unmet_targets=true", window.findChild(QLabel, "evaluationApprovalStatusLabel").text())
        self.assertEqual(service.calls, [])

        window.findChild(QLineEdit, "evaluationApprovalCriteriaEdit").setText("allow_unmet_targets=true")
        approve.click()
        deadline = time.monotonic() + 2
        while (len(service.calls) < 1 or history.rowCount() < 2) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        self.assertEqual(service.calls[0][1:3], ("approved", "engineer"))
        self.assertEqual(history.rowCount(), 2)
        self.assertEqual(history.item(1, 2).text(), "Approved")
        window.close()


if __name__ == "__main__":
    unittest.main()
