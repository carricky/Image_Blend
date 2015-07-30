#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>  
#include <CGAL/Triangulation_euclidean_traits_xy_3.h>  
#include <CGAL/Delaunay_triangulation_2.h>  

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef Delaunay::Vertex_handle Vertex_handle;

typedef K::Point_2 Point;

std::vector<K::Point_2> vertices;

int global_w, global_h;
int tri_state = 0;

//void points_draw()
//{
//	glClear(GL_COLOR_BUFFER_BIT);
//	glPushMatrix();
//
//	std::vector <K::Point_2>::iterator iter;
//	glColor3f(1.0, 1.0, 1.0);
//	glPointSize(5);
//	glBegin(GL_POINTS);
//	for (iter = vertices.begin(); iter != vertices.end(); iter++)
//		glVertex2i(iter->hx(), iter->hy());
//	glEnd();
//
//	glPopMatrix();
//	glutSwapBuffers();
//}

//void points_add_point(int x, int y)
//{
//	vertices.push_back(K::Point_2(x, global_h - y));
//}

//
//void points_clear()
//{
//	glClear(GL_COLOR_BUFFER_BIT);
//	glPushMatrix();
//	glPopMatrix();
//	glutSwapBuffers();
//
//	vertices.clear();
//	tri_state = 0;
//}
//
//void read_file()//从文件中读入点集数据，调试时所使用  
//{
//	FILE* f;
//	f = freopen("data.txt", "r", stdin);
//
//	int a, b;
//	while (std::cin >> a >> b)
//	{
//		vertices.push_back(K::Point_2(a, b));
//	}
//
//	fclose(f);
//}

void points_triangulation()
{
	Delaunay dt;//Delaunay数据结构，代表当前数据的一个且仅有一个的三角剖分，详情请参考CGAL_manual  

	dt.insert(vertices.begin(), vertices.end());//输入数据  

	//points_draw();//points_draw()函数中已经调用一次glutSwapBuffers()，本函数再一次调用glutSwapBuffers()  
	//在一帧的绘制中两次调用glutSwapBuffers()，虽对本例无影响，但存在一些问题，这不是本文的重点，可暂且忽略之  

	//glPushMatrix();

	//Delaunay::Finite_faces_iterator fit;//遍历Delaunay的所有面（有限面），将每个面的边画出来  
	//glColor3f(0.0, 0.0, 1.0);
	//for (fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); fit++)
	//{
	//	glBegin(GL_LINE_LOOP);
	//	glVertex2i(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy());
	//	glVertex2i(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy());
	//	glVertex2i(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy());
	//	glEnd();
	//}//完成Delaunay三角剖分的绘制，Delaunay图  

	//Delaunay::Edge_iterator eit;//遍历Delaunay的所有边，绘制Delaunay图的对偶图，即Voronoi图  

	//glEnable(GL_LINE_STIPPLE);//使用点画模式，即使用虚线来绘制Voronoi图  
	//glLineStipple(1, 0x3333);
	//glColor3f(0.0, 1.0, 0.0);

	//for (eit = dt.edges_begin(); eit != dt.edges_end(); eit++)
	//{
	//	CGAL::Object o = dt.dual(eit);//边eit在其对偶图中所对应的边  

	//	if (CGAL::object_cast<K::Segment_2>(&o)) //如果这条边是线段，则绘制线段  
	//	{
	//		glBegin(GL_LINES);
	//		glVertex2i(CGAL::object_cast<K::Segment_2>(&o)->source().hx(), CGAL::object_cast<K::Segment_2>(&o)->source().hy());
	//		glVertex2i(CGAL::object_cast<K::Segment_2>(&o)->target().hx(), CGAL::object_cast<K::Segment_2>(&o)->target().hy());
	//		glEnd();
	//	}
	//	else if (CGAL::object_cast<K::Ray_2>(&o))//如果这条边是射线，则绘制射线  
	//	{
	//		glBegin(GL_LINES);
	//		glVertex2i(CGAL::object_cast<K::Ray_2>(&o)->source().hx(), CGAL::object_cast<K::Ray_2>(&o)->source().hy());
	//		glVertex2i(CGAL::object_cast<K::Ray_2>(&o)->point(1).hx(), CGAL::object_cast<K::Ray_2>(&o)->point(1).hy());
	//		glEnd();
	//	}
	//}
	//glDisable(GL_LINE_STIPPLE);//关闭点画模式  

	/*glPopMatrix();
	glutSwapBuffers();*/

	tri_state = 1;//完成三角剖分，置状态为1  
}

//void display(void)
//{
//}
//
//void init(void)
//{
//	glClearColor(0.0, 0.0, 0.0, 0.0);
//	glShadeModel(GL_FLAT);
//
//}

//void reshape(int w, int h)
//{
//	global_w = w;
//	global_h = h;
//	points_clear();
//
//	glViewport(0, 0, w, h);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//
//	glOrtho(0, w, 0, h, -1.0, 1.0);
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//}
//
//void mouse(int button, int state, int x, int y)
//{
//	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
//	{
//		if (tri_state == 1) points_clear();
//		else
//		{
//			points_add_point(x, y);
//			//read_file();  
//			points_draw();
//		}
//	}
//	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
//	{
//		if (tri_state == 1) points_clear();
//		else points_triangulation();
//	}
//}
//
//void keyboard(unsigned char key, int x, int y)
//{
//	switch (key)
//	{
//	case 27:
//		exit(0);
//		break;
//	}
//}

//int main(int argc, char** argv)
//{
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
//	glutInitWindowSize(800, 600);
//	glutInitWindowPosition(100, 100);
//	glutCreateWindow(argv[0]);
//	init();
//	glutDisplayFunc(display);
//	glutReshapeFunc(reshape);
//	glutMouseFunc(mouse);
//	glutKeyboardFunc(keyboard);
//	glutMainLoop();
//	return 0;
//}