import requests
import json
from box import Box
# URL của API
url = "https://apiquanlydaotao.ptit.edu.vn/chi-tieu/thi-sinh/co-so/BVH?nam=2024"

# Gửi yêu cầu GET đến API
response = requests.get(url)

# Kiểm tra nếu yêu cầu thành công
if response.status_code == 200:
    # Phân tích JSON từ phản hồi
    data = response.json()

    # Lưu trữ dữ liệu vào file JSON
    with open('data/tuyensinh_2024/chi_tieu_BVH_ver2.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Dữ liệu đã được lưu vào file.")
else:
    print(f"Yêu cầu không thành công. Mã lỗi: {response.status_code}")


# # Đọc file JSON vào một Box object
# with open('chi_tieu_BVH.json', 'r', encoding='utf-8') as file:
#     data = Box(json.load(file))

# print(data.data[0].maNganh)
