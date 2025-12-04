Reports
========================
To view quality control reports online, you can run MEGPrep with Docker using the following command:

.. code-block:: bash

    docker run -p 8501:8501 -v /data/liaopan/datasets/SMN4Lang/g:/output cmrlab/megprep:<version> -r

This command maps the report server port and specifies the output directory for the generated reports.

**Access via browser**

.. code-block:: bash

    http://<server_ip>:8501