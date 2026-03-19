import urllib.request
urls = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.XPT",
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.XPT",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.xpt"
]
for u in urls:
    req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        r = urllib.request.urlopen(req)
        d = r.read(100)
        if b"Page Not Found" not in d and b"<html" not in d.lower()[:50]:
            print("OK", u)
        else:
            print("404 (HTML)", u)
    except Exception as e:
        print("ERR", u, e)
