import json
from bs4 import BeautifulSoup
import requests
nganh = {

}

nganh['overview'] = []
ma = [
    "7480201_UDU",
    "7480101",
    "7480201",
    "7480202",
    "7520207",
    "7510301",
    "7329001",
    "7320104",
    "7340115",
    "7340122",
    "7340101",
    "7340301",
    "7340202",
    "7480201(CLC)",
    "7340208",
    "7480205",
    "7520216",
    "7340301 (2022)",
    "7320101",
    "7340205",
    "7480102",
    "7340115_CLC",
    "7520208",
    "734030"
]
# for i in range(len(maNganhDaoTao)):
# URL của trang web cần crawl
base_url = "https://daotao.ptit.edu.vn/nganhhoc"
for ma in ma:
    url = f"{base_url}/{ma}"
    # Gửi yêu cầu GET đến trang web và lấy nội dung HTML
    response = requests.get(url)
    html_content = response.text

    # Phân tích cú pháp HTML với BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Tìm tất cả các thẻ div có class="product"
    majors = soup.find_all('div', id='content')

    # Duyệt qua từng sản phẩm và trích xuất thông tin
    for major in majors:
        title = major.find(
            'div', class_='nganhhocstyle__Title-sc-6vsl0t-2 zCcGA').text.strip()
        nganh['titleMajor'] = title

        cards = major.find_all(
            'div', class_='Cardstyle__Content-yt1j6c-3 dJJwOZ')
        maNganh = cards[0].text.strip()
        nganh['maNganh'] = maNganh

        tgianHoc = cards[1].text.strip()
        nganh['thoigianHoc'] = tgianHoc
        kyNhapHoc = cards[2].text.strip()
        nganh['kyNhapHoc'] = kyNhapHoc

        coSo = cards[3].text.strip()
        nganh['coSo'] = coSo

        hocphis = major.find_all(
            'div', class_='CardHTMLstyle__WrapperCard-sc-17bzkgf-0 itsfsF')

        for i in range(len(hocphis)-1):
            if (i == len(hocphis)-2):
                nganh['hocPhi'] = hocphis[i].text.strip()

        overviewMajor = major.find(
            'div', class_='CardHTMLstyle__WrapperCard-sc-17bzkgf-0 kjSCyT')
        nganh["overview"] = overviewMajor.text.strip()
        # spans_in_div = overviewMajor.find_all('span')
        # for span in spans_in_div:
        #     text = span.text.strip()
        #     nganh['overview'].append(text)
        #     print(text)
        #     # print(nganh)
    file_name = f'tong_quan_ve_nganh/{ma}.json'
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(nganh, json_file, ensure_ascii=False, indent=4)
