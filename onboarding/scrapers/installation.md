We utilize various packages for our scrapers. This document serves to guide their installation on your system.

# Method 1 (Recommended): Using the *requirements.txt* file
1. Find the `requirements.txt` file located at the top level directory.
2. Then, in that same directory, run `pip3 install -r requirements.txt`

This method, however, does not install ChromeDriver used as part of Selenium because it is independent of pip. [More details.](#chromedriver-for-selenium)

# Method 2: Installing the scraping packages individually
## Pip Packages (Requests, Selenium, BeautifulSoup)
In your virtualenv, run `pip3 install requests && pip3 install selenium && pip3 install bs4`.

## ChromeDriver for Selenium
Chrome is used as it supports headless browsing, i.e. an instance of Chrome does not need to be opened every time a webpage is queried. Installation: 
1. Navigate to http://chromedriver.chromium.org/downloads
2. Follow the download link for the latest version of ChromeDriver.
3. Download and unzip the appropriate zip file for your system.
4. Add the resulting executable to your PATH (/usr/local/bin is recommended)
