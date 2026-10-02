# BMDC_costim

In the BMDC_costim folder, I used a Python script most resembling
figures_BMDC_activation_GitHub.py for Figure 5.2 in my dissertation. To
make it more usable, a V2 (figures_BMDC_activation_v2_GitHub.py) was made
where all the different hard-coded paths are up front instead of in five
separate places. BMDCs from two mice were used in Exp4, so I know mice can
share an Exp name as long as they have a separate folder. If one was to run
the script with BMDCs from the same mouse across different experiments, it
breaks. One file folder per mouse. Well IDs can match between mice (both
Exp4 mice have an A8, for example) because each mouse has its own folder,
but within one folder each well ID must appear only once: if two files
start with the same well ID (e.g. "A8 sample.fcs" and "A8 rerun.fcs"),
the script does not warn you, it just counts that well twice. If you
have this specific use case, I do not want to steal the joy of
troubleshooting code from you.
