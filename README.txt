NickyB-Flow: SPECTRAL FLOW CYTOMETRY PIPELINE
=============================================
by Nicholas B. Franks, PhD

Python scripts that take raw spectral flow cytometry files (Cytek Aurora)
through FSC/SSC gating, spectral unmixing, marker gating and gating QC, to
the final figures for two assays: dendritic-cell (BMDC) activation markers
and OT-II T-cell proliferation in response to purified lipid A fractions
(bacterial strain x thin-layer chromatography Rf band).

These scripts were used to generate the flow cytometry figures in Chapter 5
of my PhD dissertation. You can find the dissertation through the Tufts
University Tisch Library by searching:

    Nicholas Franks, Impact of Microbial Modification of Lipid A on
    Dendritic Cell Functional Responses

They are shared as working templates: with small edits (marked "EDIT FOR
YOUR DATA", or collected in a "USER SETTINGS" block at the top of a script)
they can be run on similar data from other labs. No data files are included.


Folders
-------
unmix_gate_view/   FSC/SSC gating, spectral unmixing, an interactive
                   gate-drawing viewer (Flow_View_GitHub.py), sequential
                   marker gating and per-well gating QC PDFs.
                   Start here: it produces the gated folders the other two
                   use.
BMDC_costim/       Dendritic-cell activation-marker figures (CD40, CD80,
                   CD86, PDL1) + statistics.
DDI_Tcell/         T-cell proliferation figures (CellTrace Violet dilution,
                   Density Difference Integral) + statistics.

Each folder contains
  requirements.txt   Python packages (versions used for the original figures)
  workflow.txt       what each script does, the order to run them in, the
                     expected input format, and what to change for your data
  *_GitHub.py        the scripts
BMDC_costim/ also has a short README.md of its own.


Getting started
---------------
Python 3.14 was used. For any folder:

    cd unmix_gate_view             (for example)
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/python <script>.py --help

Then follow that folder's workflow.txt.


Reading/writing .fcs files
--------------------------
Every script here reads and writes .fcs files with a small pure-Python
reader/writer (numpy + pandas only; no FlowIO/FlowKit). The same code is
available on its own as FCS_Reader_Writer:
    https://github.com/NickyB-cloud/FCS_Reader_Writer
    https://doi.org/10.5281/zenodo.23087328


License
-------
MIT (see LICENSE). Free to use, modify and share; keep the copyright notice.
