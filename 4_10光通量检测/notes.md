### 配光报告
| B 50L | Min | Max | E[lx] | H[。] | V[。] |   
| HV | Min | Max | E[lx] | H[。] | V[。] |   
| BR | Min | Max | E[lx] | H[。] | V[。] |   
| BRR | Min | Max | E[lx] | H[。] | V[。] |   
| BLL | Min | Max | E[lx] | H[。] | V[。] |   
| P | Min | Max | E[lx] | H[。] | V[。] |   
| ECE-Zone III | Min | Max | E[lx] | H[。] | V[。] |   
| S50 | Min | Max | E[lx] | H[。] | V[。] |   
| S50 LL | Min | Max | E[lx] | H[。] | V[。] |   
| S50 RR | Min | Max | E[lx] | H[。] | V[。] |   
| S100 | Min | Max | E[lx] | H[。] | V[。] |   
| S100 LL | Min | Max | E[lx] | H[。] | V[。] |   
| S100 RR | Min | Max | E[lx] | H[。] | V[。] |   
| 75R | Min | Max | E[lx] | H[。] | V[。] |   
| 50V | Min | Max | E[lx] | H[。] | V[。] |   
| 50L | Min | Max | E[lx] | H[。] | V[。] |   
| 25 LL | Min | Max | E[lx] | H[。] | V[。] |   
| 25 RR | Min | Max | E[lx] | H[。] | V[。] |   

### 算法步骤  
1. 提取远点位置近横轴，建立直角坐标系。
p1(x1, y1), p0(x0, y0), 以p0为坐标原点，p1指向p0的方向作为x轴正方向，
y轴正方向为x轴正方向逆时针旋转90度的方向，建立新的直角坐标系，
返回老坐标系和新坐标系的映射关系，使得老坐坐标系中的点能够映射到新的坐标系中。


2. 获取配光报告对应点位的亮度值
3. 建立对应点位的亮度-照度映射关系
4. 交叉验证