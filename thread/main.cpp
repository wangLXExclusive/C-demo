#include <iostream>
#include <thread>
#include <windows.h>
using namespace std;

//线程.detach  分离|线程后台运行|发后即忘
//线程.join     等待线程结束
//线程.joinable 判断线程是否join 没有的话返回true
namespace oop {
    void do_something(int i) {
        cout << i << endl;
    }

    struct Func {
        int &i;

        Func(int &i_) : i(i_) {}

        void operator()() {
            for (int j = 0; j < 10; ++j)do_something(i);
        }

    };

    void oop() {
        int some_local_state = 0;
        Func my_func(some_local_state);
        thread t(my_func);
        //t.detach();  //分离  后台运行线程
        t.join();  //等待
    }
}
namespace helloWorld {
    void hello(int i) {
        thread hello([i]() { cout << "hello" << i << endl; });
        hello.detach();
    }

    void world(int i) {
        thread world([i]() { cout << "world" << i << endl; });
        world.detach();
    }

    void f() {
        for (int i = 0; i < 10; ++i) {
            hello(i);
            world(i);
        }
    }
}
namespace thread_guard{
    class thread_guard{
        thread& t;

    public:
        explicit thread_guard(thread& t_):t(t_){}
        ~thread_guard(){
            if(t.joinable()){//判断线程是否加入
                t.join();       //没有的话进行加入操作,对线程进行等待
            }
        }

    public:
        //删除函数，是为了让编译器不要自动生成它
        //因为直接对一个对象进行拷贝和复制是十分危险的
        thread_guard(thread_guard const &)=delete ;
        thread_guard& operator=(thread_guard const &)=delete ;
    };

    void f(){
        int some_local_state=1;
        oop::Func my_func(some_local_state);
        thread t(my_func);
        thread_guard g(t);
    }//当函数进行到这里的时候
    //局部对象会被逆序销毁
    //thread_guard是第一个被销毁的，这时线程所在的析构函数运行
}
int main() {
    //oop::oop();
    //helloWorld::f();
    thread_guard::f();

    Sleep(100);
    return 0;
}
