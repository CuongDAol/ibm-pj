# Dự án SpaceX - Phân tích và Dự đoán

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
  - [Module 1: Data Collection API](#module-1-data-collection-api)
  - [Module 1: SpaceX Data Wrangling](#module-1-spacex-data-wrangling)
  - [Module 1: Web Scraping](#module-1-web-scraping)
  - [Module 2: Data Visualization](#module-2-edatadviz)
  - [Module 2: SQL Lab](#module-2-sql-lab)
  - [Module 3: Dash Interactive](#module-3-dash-interactive)
  - [Module 3: Folium Launch Site Location](#module-3-folium-launch-site-location)
  - [Module 4: SpaceX Machine Learning Prediction](#module-4-spacex-machine-learning-prediction)
- [Tác giả](#tác-giả)

## Giới thiệu

Dự án SpaceX này bao gồm nhiều module cho phép bạn thu thập, xử lý và phân tích dữ liệu từ SpaceX. Mục tiêu của dự án là phân tích thông tin về các lần phóng của SpaceX, trực quan hóa dữ liệu và áp dụng các mô hình học máy để dự đoán khả năng hạ cánh thành công của Falcon 9.

## Cấu trúc dự án

### **Module 1: Data Collection API**

**Mục tiêu**: Thu thập dữ liệu từ SpaceX API.

Đoạn mã sau sử dụng thư viện `requests` để gửi yêu cầu GET đến API SpaceX và thu thập dữ liệu về các lần phóng:

```python
import requests
import pandas as pd

# Gửi yêu cầu GET đến API SpaceX
url = 'https://api.spacexdata.com/v4/launches'  # URL của API SpaceX cho thông tin phóng

response = requests.get(url)

# Kiểm tra nếu yêu cầu thành công
if response.status_code == 200:
    data = response.json()  # Chuyển đổi dữ liệu JSON từ API thành đối tượng Python
    print(f"Data fetched successfully: {len(data)} entries")  # In ra số lượng dữ liệu
else:
    print(f"Failed to retrieve data, status code: {response.status_code}")
```
Giải thích:

Đoạn mã này kết nối đến API SpaceX và thu thập dữ liệu về các lần phóng của Falcon 9.

Kiểm tra mã trạng thái HTTP (status code) để xác nhận yêu cầu thành công.

Dữ liệu thu thập được từ API sẽ được chuyển thành định dạng JSON và sau đó lưu vào một đối tượng Python.
# Module 1: SpaceX Data Wrangling
Mục tiêu: Làm sạch và xử lý dữ liệu thu thập từ API.

Đoạn mã này sử dụng thư viện pandas để làm sạch dữ liệu và chuẩn hóa dữ liệu, bao gồm xử lý giá trị thiếu:

```python
# Kiểm tra các giá trị thiếu trong DataFrame
missing_data = launch_df.isnull().sum()
print(missing_data)

# Loại bỏ các cột không cần thiết
launch_df_cleaned = launch_df.drop(columns=['name', 'links', 'details', 'static_fire_date_utc'])

# Điền giá trị thiếu cho các cột có thể
launch_df_cleaned['date_utc'] = pd.to_datetime(launch_df_cleaned['date_utc'], errors='coerce')
launch_df_cleaned['success'] = launch_df_cleaned['success'].fillna(False)

# Kiểm tra lại dữ liệu đã được làm sạch
launch_df_cleaned.head()
```
Giải thích:

Đoạn mã kiểm tra các giá trị thiếu (null) trong dữ liệu.

Loại bỏ các cột không cần thiết để tập trung vào những thông tin quan trọng.

Điền các giá trị thiếu trong cột success bằng False và chuyển đổi các giá trị trong cột date_utc thành định dạng ngày tháng (datetime).

# Module 1: Web Scraping
Mục tiêu: Thu thập dữ liệu từ trang web của SpaceX bằng kỹ thuật Web Scraping.

```python
import requests
from bs4 import BeautifulSoup

# Gửi yêu cầu GET đến trang web SpaceX
url = 'https://www.spacex.com/launches/'
response = requests.get(url)

# Phân tích dữ liệu HTML từ trang web
soup = BeautifulSoup(response.content, 'html.parser')

# Trích xuất các thông tin quan trọng như tên phóng, thời gian phóng
launches = soup.find_all('div', class_='launch-card')
for launch in launches:
    name = launch.find('h3').text
    date = launch.find('time').text
    print(f"Launch: {name}, Date: {date}")
```
Giải thích:

Đoạn mã sử dụng thư viện requests để lấy dữ liệu HTML từ trang web SpaceX.

BeautifulSoup giúp phân tích và trích xuất thông tin về các lần phóng từ HTML, chẳng hạn như tên và thời gian phóng.

# Module 2: Data Visualization
Mục tiêu: Trực quan hóa dữ liệu để phân tích các xu hướng và mẫu.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Vẽ biểu đồ cột cho doanh thu theo tháng
monthly_sales_df = all_data.groupby(['Month']).sum()
months = range(1, 13)

plt.bar(months, monthly_sales_df['Sales']/1000000)
plt.xticks(months)
plt.ylabel('Sales in USD ($ Million)')
plt.xlabel('Month number')
plt.show()
```
Giải thích:

Đoạn mã này vẽ biểu đồ cột để thể hiện doanh thu của SpaceX theo từng tháng.

Dữ liệu doanh thu đã được nhóm theo tháng và sau đó biểu diễn dưới dạng cột.

# Module 2: SQL Lab
Mục tiêu: Quản lý và truy vấn dữ liệu sử dụng cơ sở dữ liệu SQL.
```python
import sqlite3
import pandas as pd

# Kết nối với cơ sở dữ liệu SQLite
conn = sqlite3.connect('spacex.db')

# Thực hiện truy vấn SQL
query = "SELECT * FROM launches WHERE success = 1"
successful_launches = pd.read_sql(query, conn)

# Hiển thị kết quả
print(successful_launches.head())
```
Giải thích:

Đoạn mã kết nối đến cơ sở dữ liệu SQLite và thực hiện một truy vấn SQL để lấy các lần phóng thành công từ bảng launches.

Dữ liệu được đọc và lưu vào một DataFrame của pandas để dễ dàng xử lý và phân tích.

# Module 3: Dash Interactive
Mục tiêu: Tạo một ứng dụng web tương tác sử dụng Dash để hiển thị dữ liệu SpaceX.

```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# Tạo ứng dụng Dash
app = dash.Dash(__name__)

# Tạo một biểu đồ phân phối của các lần phóng thành công
df = pd.read_csv('spacex_data.csv')
fig = px.histogram(df, x="mission", color="success")

# Định nghĩa layout của ứng dụng
app.layout = html.Div([
    html.H1("SpaceX Mission Success"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
```
Giải thích:

Đoạn mã này tạo một ứng dụng Dash để hiển thị biểu đồ phân phối các lần phóng thành công của SpaceX.

Biểu đồ được tạo bằng plotly.express và hiển thị trong ứng dụng web Dash.

# Module 3: Folium Launch Site Location
Mục tiêu: Trực quan hóa các địa điểm phóng SpaceX trên bản đồ.

```python
import folium

# Tạo bản đồ với vị trí phóng
m = folium.Map(location=[34.05, -118.25], zoom_start=10)

# Đánh dấu vị trí phóng trên bản đồ
folium.Marker([34.05, -118.25], popup='SpaceX Launch Site').add_to(m)

# Lưu bản đồ vào tệp HTML
m.save('launch_site_map.html')
```
Giải thích:

Đoạn mã này sử dụng thư viện folium để tạo một bản đồ hiển thị các địa điểm phóng của SpaceX.

Một đánh dấu được thêm vào bản đồ để thể hiện vị trí phóng và bản đồ sẽ được lưu dưới dạng tệp HTML.

# Module 4: SpaceX Machine Learning Prediction
Mục tiêu: Dự đoán khả năng hạ cánh thành công của Falcon 9 bằng học máy.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Chuẩn bị dữ liệu
X = all_data[['feature1', 'feature2', 'feature3']]  # Các đặc trưng
y = all_data['success']  # Nhãn hạ cánh thành công

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```
Giải thích:

Đoạn mã này sử dụng mô hình Random Forest để dự đoán khả năng hạ cánh thành công của Falcon 9.

Dữ liệu được chia thành tập huấn luyện và kiểm tra, và mô hình được huấn luyện và đánh giá độ chính xác.

## Tác giả
Dự án này được thực hiện bởi CuongDAol.

Email: buipg0801@gmai.com

Cảm ơn bạn đã xem qua dự án này! Nếu bạn có bất kỳ câu hỏi nào hoặc muốn hợp tác, đừng ngần ngại liên hệ với tôi.
