import requests
from bs4 import BeautifulSoup
import json
import csv
# URL của trang web bạn muốn crawl
url = 'https://diemthi.tuyensinh247.com/diem-chuan/hoc-vien-cong-nghe-buu-chinh-vien-thong-phia-nam-BVS.html'

# Gửi yêu cầu GET tới trang web
response = requests.get(url)

# Kiểm tra xem yêu cầu có thành công hay không
if response.status_code == 200:
    # Lấy nội dung của trang web
    page_content = response.content

    # Phân tích nội dung bằng BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Tìm thẻ div có id = 'tab_1'
    div_tab_1 = soup.find('div', {'id': 'tab_1'})

    if div_tab_1:
        # Tìm bảng dữ liệu bên trong thẻ div này
        # Giả sử bảng là thẻ table đầu tiên trong div
        table = div_tab_1.find('table')

        if table:
            # Lấy tất cả các hàng trong bảng
            rows = table.find_all('tr')

            # Mở file CSV để ghi dữ liệu
            with open('data/score/2024/BVS/THPT.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Duyệt qua từng hàng và ghi dữ liệu vào file CSV
                for row in rows:
                    cells = row.find_all('td')
                    cell_data = [cell.text.strip() for cell in cells]
                    print(cell_data)
                    writer.writerow(cell_data)

            print("Dữ liệu đã được lưu vào file output.csv.")
        else:
            print("Không tìm thấy bảng dữ liệu trong thẻ div.")
    else:
        print("Không tìm thấy thẻ div với id='tab_1'.")
else:
    print(f"Không thể truy cập trang web. Mã lỗi: {response.status_code}")


# # dùng selenium để click button crawl score before 2024
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.keys import Keys
# from bs4 import BeautifulSoup
# import time
# import csv
# # Khởi tạo trình duyệt (chọn trình duyệt bạn muốn sử dụng)
# driver = webdriver.Chrome()

# # Mở trang web
# url = "https://diemthi.tuyensinh247.com/diem-chuan/hoc-vien-cong-nghe-buu-chinh-vien-thong-phia-nam-BVS.html"
# driver.get(url)

# # Tìm phần tử cần click
# load_button = driver.find_element(By.CLASS_NAME, 'load_method_6')

# # Scroll đến phần tử
# actions = ActionChains(driver)
# actions.move_to_element(load_button).perform()

# # Đợi một chút để đảm bảo phần tử đã hiển thị
# time.sleep(1)
# # Thử click vào phần tử
# try:
#     load_button.click()
# except Exception as e:
#     print(f"Lỗi khi click: {e}")

# # Đợi một chút để dữ liệu tải
# time.sleep(5)
# # Tìm phần tử cần click
# load_button = driver.find_element(By.CLASS_NAME, 'load_method_6')

# # Scroll đến phần tử
# actions = ActionChains(driver)
# actions.move_to_element(load_button).perform()

# # Đợi một chút để đảm bảo phần tử đã hiển thị
# time.sleep(1)

# # Thử click vào phần tử
# try:
#     load_button.click()
# except Exception as e:
#     print(f"Lỗi khi click: {e}")

# # Đợi một chút để dữ liệu tải
# time.sleep(5)

# # Lấy nội dung trang sau khi click
# page_source = driver.page_source

# # Phân tích nội dung trang bằng BeautifulSoup
# soup = BeautifulSoup(page_source, 'html.parser')

# # Tìm div có class 'more_data_container_1'
# more_data_container = soup.find('div', id='more_data_container_6')


# # Giả sử trong div này có nhiều bảng, lấy bảng thứ 2 (index 1 trong Python)
# tables = more_data_container.find_all('table')
# table = tables[1]
# if table:
#     # Lấy tất cả các hàng trong bảng
#     rows = table.find_all('tr')

#     # Mở file CSV để ghi dữ liệu
#     with open('data/score/2022/BVS/DGTD.csv', 'w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)

#         # Duyệt qua từng hàng và ghi dữ liệu vào file CSV
#         for row in rows:
#             cells = row.find_all('td')
#             cell_data = [cell.text.strip() for cell in cells]
#             writer.writerow(cell_data)
#             print(cell_data)

#     print("Dữ liệu đã được lưu vào file output.csv.")
# else:
#     print("Không tìm thấy bảng dữ liệu trong thẻ div.")

# # Đóng trình duyệt
# driver.quit()

# crawl pt XTKH
# import requests
# from bs4 import BeautifulSoup
# import csv
# # URL của trang web bạn muốn crawl
# url = 'https://thptquocgia.org/diem-chuan-xet-tuyen-ket-hop-hv-cong-nghe-buu-chinh-vien-thong-nam-2021'

# # Gửi yêu cầu GET tới trang web
# response = requests.get(url)

# # Kiểm tra xem yêu cầu có thành công hay không
# if response.status_code == 200:
#     # Lấy nội dung của trang web
#     page_content = response.content

#     # Phân tích nội dung bằng BeautifulSoup
#     soup = BeautifulSoup(page_content, 'html.parser')

#     # Tìm thẻ div có id = 'tab_1'
#     div_tab_1 = soup.find('div', {'class': 'entry-content single-page'})

#     if div_tab_1:
#         # Tìm bảng dữ liệu bên trong thẻ div này
#         # Giả sử bảng là thẻ table đầu tiên trong div
#         tables = div_tab_1.find_all('table')
#         table = tables[0]

#         if table:
#             # Lấy tất cả các hàng trong bảng
#             rows = table.find_all('tr')

#             # Mở file CSV để ghi dữ liệu
#             with open('data/score/2021/BVH/XTKH.csv', 'w', newline='', encoding='utf-8') as file:
#                 writer = csv.writer(file)

#                 # Duyệt qua từng hàng và ghi dữ liệu vào file CSV
#                 for row in rows:
#                     cells = row.find_all('td')
#                     cell_data = [cell.text.strip() for cell in cells]
#                     print(cell_data)
#                     writer.writerow(cell_data)

#             print("Dữ liệu đã được lưu vào file output.csv.")
#         else:
#             print("Không tìm thấy bảng dữ liệu trong thẻ div.")
#     else:
#         print("Không tìm thấy thẻ div với id='tab_1'.")
# else:
#     print(f"Không thể truy cập trang web. Mã lỗi: {response.status_code}")
