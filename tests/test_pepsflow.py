from pepsflow.pepsflow import Pepsflow
from pepsflow.pepsflow import IO
from pepsflow.ipeps.observe import Observer

import pytest


class TestPepsflow:
    def test_general_workflow_minimal_canonical(self):
        """
        Test the general workflow of the Pepsflow class with minimal canonical form.
        """
        workflow = Pepsflow(config_path="tests/test_cfgs/test_pepsflow1.cfg")

        # Optimize the iPEPS state in the Heisenberg model for chi=8, D=3
        workflow.optimize_parallel()

        # Gauge transform the iPEPS state
        workflow.gauge("D3_test")

        # Evaluate the iPEPS state at chi=16
        workflow.evaluate("D3_test_minimal_canonical")

        ipeps = IO.load("tests/test_data/D3_test_minimal_canonical")
        Energy = Observer(ipeps).eval_energies()[-1]
        assert Energy == pytest.approx(-0.6681273941483516, abs=1e-3)

    def test_general_workflow_invertible_gauge(self):
        """
        Test the general workflow of the Pepsflow class with an invertible gauge transformation.
        """
        workflow = Pepsflow(config_path="tests/test_cfgs/test_pepsflow2.cfg")
        workflow.optimize_parallel()
        workflow.gauge("D3_test")
        workflow.evaluate("D3_test_invertible_gauge_seed5")
        ipeps = IO.load("tests/test_data/D3_test_invertible_gauge_seed5")
        Energy = Observer(ipeps).eval_energies()[-1]
        assert Energy == pytest.approx(-0.6681273941483516, abs=1e-3)
