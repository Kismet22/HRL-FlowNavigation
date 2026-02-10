class Pinball {
  BDIM flow;
  BodyUnion body;
  EllipseBody ellipse; // 保存椭圆体引用
  boolean QUICK = true, order2 = true;
  int n, m, out, up, resolution, NT = 1;
  float dt, t, D, xi0, xi1, xi2, theta;
  float xi0_m, xi1_m, xi2_m, gR, theta_m, r, dphi0, dphi1, dphi2;
  float pos_x_default, pos_y_default;
  FloodPlot flood;
  PVector force, force_0, force_1, force_2;
  PVector vel, pos;
  ArrayList<Float> surfacePressures;
  ForceInfo ellipseForceInfo;
  float x1, y1, x2, y2, x3, y3; // 圆形体位置

  // 构造函数
  Pinball(int resolution, int Re, float gR, float theta, float xi0, float xi1, float xi2,
          float dtReal, int xLengths, int yLengths, boolean isResume,
          float _pos_x, float _pos_y, float simNum) {

    n = xLengths * resolution;
    m = yLengths * resolution;

    pos_x_default = (_pos_x != 0.0f || _pos_y != 0.0f) ? _pos_x : 6 * n / 8;
    pos_y_default = (_pos_x != 0.0f || _pos_y != 0.0f) ? _pos_y : m / 2;

    this.resolution = resolution;
    this.xi0 = xi0;
    this.xi1 = xi1;
    this.xi2 = xi2;
    this.gR = gR;
    this.theta = theta;
    this.dt = dtReal * this.resolution;
    theta_m = theta;

    Window view = new Window(0, 0, n, m);
    D = resolution;

    float r = D + gR * D;
    int simNumInt = (int) simNum;

    // 更改圆柱摆放位置
    if (isResume) {
      String[] lines = loadStrings("saved/init/init_" + str(simNumInt) + ".txt");
      x1 = float(lines[0]);
      y1 = float(lines[1]);
      x2 = float(lines[2]);
      y2 = float(lines[3]);
      x3 = float(lines[4]);
      y3 = float(lines[5]);
    } else {
      x1 = n / 6;
      y1 = m / 2 + r / 2;
      x2 = n / 6;
      y2 = m / 2 - r / 2;
      x3 = n / 6 + r * cos(theta);
      y3 = m / 2;
    }

    // 创建椭圆体，并保存引用
    ellipse = new EllipseBody(pos_x_default, pos_y_default, D/2, 1.5, view);

    // 创建 BodyUnion，圆形体 + 椭圆体
    body = new BodyUnion(
      new CircleBody(x1, y1, D, view),
      new CircleBody(x2, y2, D, view),
      new CircleBody(x3, y3, D, view),
      ellipse
    );

    flow = new BDIM(n, m, dt, body, (float) D / Re, QUICK);

    if (isResume) {
      flow.resume("saved/init/init_" + str(simNumInt) + ".bdim");
    }

    flood = new FloodPlot(view);
    flood.range = new Scale(-1, 1);
    flood.setLegend("vorticity");

    force_0 = new PVector(); // 初始化力
  }

  // ----------------------------
  // 单独修改椭圆体速度
  void setEllipseVelocity(PVector vel) {
      if (ellipse != null) {
          println("Setting ellipse velocity to: " + vel);
          ellipse.setVelocity(vel);
      } else {
          println("Ellipse is null, cannot set velocity!");
      }
  }

  // ----------------------------
  // 更新仿真
  // 外部作用力 + 流体作用力
  void update2() {
    flow.dt = dt;
    force_0.x = xi0;
    force_0.y = xi1;
    dphi0 = xi2; // torque

    // 椭圆体响应
    // body.bodyList.get(3).react(flow, force_0, dphi0);
    // 椭圆力已在外部定义
    ellipseForceInfo = body.bodyList.get(3).react_1(flow, force_0, dphi0);


    // 更新流场
    flow.update(body);
    if (order2) {
      flow.update2(body);
    }

    t += dt / resolution;

    vel = body.bodyList.get(3).dotxc;
    vel.z = body.bodyList.get(3).dotphi;
    pos = body.bodyList.get(3).xc;
    pos.z = body.bodyList.get(3).phi;

    // 更新表面压力
    surfacePressures = body.bodyList.get(3).calculateSurfacePressures(flow.p);
  }

  // ----------------------------
  void display() {
    flood.display(flow.u.curl());
    body.display();
    flood.displayTime(t);
  }
}



// class Pinball {
//   BDIM flow;
//   BodyUnion body;
//   boolean QUICK = true, order2 = true;
//   int n, m, out, up, resolution,NT=1;
//   float dt, t, D, xi0, xi1, xi2,theta;
//   float xi0_m, xi1_m, xi2_m, gR, theta_m, r, dphi0, dphi1, dphi2;
//   float pos_x_default, pos_y_default;
//   FloodPlot flood;
//   PVector force, force_0, force_1, force_2;
//   PVector vel, pos;
//   ArrayList<Float> surfacePressures;
//   float x1, y1, x2, y2, x3, y3; // TODO: 定义变量

// // 新增Float:vx vy,允许从外部传入一个初始速度,可以为空
//   //Pinball (int resolution, int Re,  float gR,  float theta,  float xi0,  float xi1,  float xi2, float dtReal, int xLengths, int yLengths, boolean isResume, float _pos_x, float _pos_y, float simNum, Float initVx, Float initVy) {
//   Pinball (int resolution, int Re,  float gR,  float theta,  float xi0,  float xi1,  float xi2, float dtReal, int xLengths, int yLengths, boolean isResume, float _pos_x, float _pos_y, float simNum) {
//     // resolution:理解成缩放倍数
//     // resolution取16
//     n = xLengths*resolution;
//     m = yLengths*resolution;

//     // 默认速度为 0，如果传 null 则取 0
//     // float vx = (initVx != null) ? initVx : 0.0;
//     // float vy = (initVy != null) ? initVy : 0.0;
//     // PVector initVel = new PVector(vx, vy);

//     if (_pos_x != 0.0f || _pos_y != 0.0f) {
//         // 不同时为0，使用传入坐标
//         pos_x_default = _pos_x;
//         pos_y_default = _pos_y;
//     } else {
//         // 否则使用默认坐标(240,64)
//         pos_x_default = 6 * n / 8;  
//         pos_y_default = m / 2;  
//     }

//     this.resolution = resolution;
//     this.xi0 = xi0;
//     this.xi1 = xi1;
//     this.xi2 = xi2;
//     this.gR = gR;
//     this.theta = theta;
//     this.dt = dtReal*this.resolution;
//     theta_m=theta;

//     Window view = new Window(0, 0, n, m); // zoom the display around the body
//     D=resolution;

//     float r=D+gR*D;
//     int simNumInt = (int)simNum;

    
//   // TODO: 读已存的角度
//     if (isResume) {
//       String[] lines = loadStrings("saved/init/init_" + str(simNumInt) + ".txt");
//       x1 = float(lines[0]);
//       y1 = float(lines[1]);
//       x2 = float(lines[2]);
//       y2 = float(lines[3]);
//       x3 = float(lines[4]);
//       y3 = float(lines[5]);
//     } 

//     // println(x1, y1, x2, y2, x3, y3);

//     // 初始化速度版本
//     // body = new BodyUnion(
//     //     new CircleBody(x1, y1, D, view),
//     //     new CircleBody(x2, y2, D, view),
//     //     new CircleBody(x3, y3, D, view),
//     //     new EllipseBody(pos_x_default, pos_y_default, D/2, 1.5, view, initVel)
//     // );

//     EllipseBody ellipse = new EllipseBody(pos_x_default, pos_y_default, D/2, 1.5, view);

//     // 同时创建了四个body实例,组成一个bodyunion
//     body = new BodyUnion(
//         new CircleBody(x1, y1, D, view),
//         new CircleBody(x2, y2, D, view),
//         new CircleBody(x3, y3, D, view),
//         ellipse
//     );

//     // body =new BodyUnion(new CircleBody(n/6, m/2+r/2, D, view),
//     // new CircleBody(n/6, m/2-r/2, D, view),
//     // new CircleBody(n/6+r*cos(theta), m/2, D, view),
//     // // 椭圆体的坐标,定义在Body.pde文件中
//     // new EllipseBody(pos_x_default, pos_y_default, D/2, 1.5, view));
//     flow = new BDIM(n,m,dt,body,(float)D/Re,QUICK);
    
//     // if(isResume){
//     //   // flow.resume("saved_1/init/init.bdim");// initial state with swimmer
//     //   flow.resume("saved_1/init/init_1.bdim");// initial state without swimmer
//     // }

//     // TODO: 读已存的流场数据
//     if(isResume){
//       flow.resume("saved/init/init_" + str(simNumInt) + ".bdim");
//     }
    
//     flood = new FloodPlot(view);
//     flood.range = new Scale(-1, 1);
//     flood.setLegend("vorticity"); 

//     force_0 = new PVector(); // 初始化 force_0

//     // 单独修改椭圆体速度
//     void setEllipseVelocity(PVector vel) {
//         if (body != null && body.bodyList.size() > 0) {
//             // 假设椭圆体在 bodyUnion 的最后一个
//             Body ellipseBody = body.bodyList.get(body.bodyList.size() - 1);
//             ellipseBody.setVelocity(vel);
//         }
//     }
//   }


// void update2(){
//   //dt = flow.checkCFL();
//   flow.dt = dt;
//   force_0.x = xi0;
//   force_0.y = xi1;
//   dphi0 = xi2; // torque, 顺时针为正

//   // body.bodyList.get(3).react(flow);
//   body.bodyList.get(3).react(flow,force_0,dphi0);
              
//   flow.update(body);
//   if (order2) {flow.update2(body);}

//   t += dt/resolution;  //nonedimension
  
//   vel = body.bodyList.get(3).dotxc;
//   vel.z = body.bodyList.get(3).dotphi;
//   pos = body.bodyList.get(3).xc;
//   pos.z = body.bodyList.get(3).phi;
//   // 计算表面压力
//   // 处理 surfacePressures 列表中的数据
//   surfacePressures = body.bodyList.get(3).calculateSurfacePressures(flow.p);

//   //flow.u中存储速度信息
//   //println("Flow velocity at target [88][64]: " + flow.u.x.a[88][64] + ", " + flow.u.y.a[88][64]);
//  }

//   void display() {
//     flood.display(flow.u.curl());
//     body.display();
//     flood.displayTime(t);
//   }
// }
