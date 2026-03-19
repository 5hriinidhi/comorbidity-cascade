import urllib.request
import re

url = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2015'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        links = re.findall(r'href=[\"\']([^\"\']*\.XPT)[\"\']', html, re.IGNORECASE)
        print("FOUND XPT LINKS:")
        for link in links:
            print(link)
except Exception as e:
    print(e)
