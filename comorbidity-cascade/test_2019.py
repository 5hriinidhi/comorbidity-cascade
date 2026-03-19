import urllib.request
import re

url = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2017'
url2 = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2019'
for u in [url, url2]:
    req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        links = set(re.findall(r'href=["\']([^"\']*\.XPT)["\']', html, re.IGNORECASE))
        print("XPT LINKS FOR", u.split('=')[-1], ":")
        for link in links:
            print(link)
    except Exception as e:
        print(e)
