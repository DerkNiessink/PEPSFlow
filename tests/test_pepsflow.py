from pepsflow.pepsflow import Pepsflow
from pepsflow.pepsflow import IO
from pepsflow.ipeps.observe import Observer
from pepsflow.models.canonize import canonize

import pytest


class TestPepsflow:
    @pytest.mark.skip()
    def test_general_workflow_minimal_canonical(self):
        """
        Test the general workflow of the Pepsflow class with minimal canonical form.
        """
        workflow = Pepsflow(config_path="tests/test_cfgs/test_pepsflow1.cfg")

        # Optimize the iPEPS state in the Heisenberg model for chi=8, D=3
        workflow.optimize_parallel()

        # Gauge transform the iPEPS state
        workflow.gauge("chi_8")

        # Evaluate the iPEPS state at chi=16
        workflow.evaluate_parallel("chi_8")

        ipeps = IO.load("tests/test_data/chi_16")

        # Check energy and norm
        Energy = Observer(ipeps).eval_energy()
        _, norm = canonize(ipeps)

        assert Energy == pytest.approx(-0.6681273941483516, abs=1e-3)
        assert norm[-1] == pytest.approx(1e-16, abs=1e-3)
        assert len(norm) == 1

    def test_general_workflow_invertible_gauge(self):
        """
        Test the general workflow of the Pepsflow class with an invertible gauge transformation.
        """
        workflow = Pepsflow(config_path="tests/test_cfgs/test_pepsflow2.cfg")
        workflow.optimize_parallel()
        workflow.gauge("chi_8")
        workflow.evaluate_parallel("chi_8")
        ipeps = IO.load("tests/test_data/chi_16")
        Energy = Observer(ipeps).eval_energy()
        assert Energy == pytest.approx(-0.6681273941483516, abs=1e-3)
