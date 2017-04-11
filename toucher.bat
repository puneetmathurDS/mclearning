FOR /F "TOKENS=1 eol=/ DELIMS=/ " %%A IN ('DATE/T') DO SET dd=%%A
FOR /F "TOKENS=1,2 eol=/ DELIMS=/ " %%A IN ('DATE/T') DO SET mm=%%B
FOR /F "TOKENS=1,2,3 eol=/ DELIMS=/ " %%A IN ('DATE/T') DO SET yyyy=%%C
SET todaysdate=%yyyy%%mm%%dd%
echo %dd%
echo %mm%
echo %yyyy%
echo %todaysdate%
SET todaysdate=%todaysdate% & ".txt"
type nul > %todaysdate%
date /t >> %todaysdate%
date /t >> %todaysdate%
date /t >> %todaysdate%
git add .
git commit -m %todaysdate%
git push origin master