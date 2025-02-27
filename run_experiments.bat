@echo off
setlocal enabledelayedexpansion

:: Define the list of sigma_ista values
set sigma_values=0.15082 0.08104 0.05619 0.04326 0.03526 0.02983 0.02590

:: Loop through each value and run the command
for %%s in (%sigma_values%) do (
    echo Running with sigma_ista=%%s
    python main.py --dp_ppr True --sigma_ista %%s
)
