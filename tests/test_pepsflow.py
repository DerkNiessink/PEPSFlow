from pepsflow.pepsflow import Pepsflow
from pepsflow.pepsflow import IO
from pepsflow.ipeps.observe import Observer
from pepsflow.models.canonize import canonize

import pytest


class TestPepsflow:
    def test_general_workflow(self):
        """
        Test the general workflow of the Pepsflow class.
        """
        workflow = Pepsflow(config_path="tests/test_pepsflow.cfg")

        # Optimize the iPEPS state in the Heisenberg model for chi=8, D=3
        workflow.optimize_parallel()

        # Gauge transform the iPEPS state
        workflow.gauge("chi_8")

        # Evaluate the iPEPS state at chi=16
        workflow.evaluate_parallel("chi_8")

        ipeps = IO.load("tests/chi_16")

        # Check energy and norm
        Energy = Observer(ipeps).eval_energy()
        _, norm = canonize(ipeps)

        assert (Energy == pytest.approx(-0.6681273941483516, abs=1e-3)) and (
            norm[-1] == pytest.approx(1e-16, abs=1e-3) and len(norm) == 1
        )
