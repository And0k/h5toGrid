:: DOS batch script to download selected files from rda.ucar.edu using Wget
::
:: Experienced Wget Users: add additional command-line flags here
::   Use the -r (--recursive) option with care
set opts=-N
::
set cert_opt=
:: If you get a certificate verification error (version 1.10 or higher),
:: uncomment the following line:
::set cert_opt=--no-check-certificate
::
:: download the file(s)
wget %cert_opt% %opts% https://request.rda.ucar.edu/dsrqst/KORZH794079/wnd10m.cdas1.202308.grb2.nc.zip
wget %cert_opt% %opts% https://request.rda.ucar.edu/dsrqst/KORZH794079/wnd10m.cdas1.202309.grb2.nc.zip
