#include <iostream>
#include <list>
#include <algorithm>

using namespace  std;


class CacheLRU{
public:
    CacheLRU(initializer_list<int> ls){
        for_each(ls.begin(),ls.end(),
                [=](const int element)
                {wait_list.push_back(element);});
    }

    void work(){
        while (!wait_list.empty()){
            if(cache.size()<cap){//如果没有超出cache
                if(find(cache.begin(),cache.end(),wait_list.front())==cache.end()) {//如果没有在cache里
                    cout<<"("<<wait_list.front()<<")"; //显示将要进入cache的值
                    cache.push_front(wait_list.front());//从头部插入
                    wait_list.pop_front();
                    print(cache);//显示cache内的值
                    cout<<"[]"<<endl;//显示淘汰值，并换行
                }else{//将当前元素移动到链表的头部
                    do_if_in_cache();
                }
            }else{//置换策略
                if(find(cache.begin(),cache.end(),wait_list.front())==cache.end()) {//如果没有在cache里
                    auto tem=cache.back(); //先保存一下cache里最后一个元素
                    cout<<"("<<wait_list.front()<<")"; //显示将要进入cache的值
                    cache.pop_back();     //删除最不常用的值
                    cache.push_front(wait_list.front());//插入等待队列的第一个值
                    wait_list.pop_front();
                    print(cache);//显示cache内的值
                    cout<<"["<<tem<<"]"<<endl; //显示淘汰值，并换行
                } else{
                    do_if_in_cache();
                }
            }
        }
    }
private:
    enum {cap=5};
    list<int> cache;      //缓存
    list<int> wait_list; //等待队列

    void print(list<int> arr){
        for_each(arr.begin(),arr.end(),
                 [=](const int &element)
                 {cout<<element<<ends;});
    }
    void do_if_in_cache(){
        auto position=find(cache.begin(),cache.end(),wait_list.front());//获得元素位置
        //移动
        auto tem=*position;
        cache.erase(position);
        cache.push_front(tem);
        //显示
        cout<<"("<<wait_list.front()<<")"; //显示将要进入cache的值
        wait_list.pop_front();
        print(cache);//显示cache内的值
        cout<<"[]"<<endl;//显示要淘汰的值，并换行
    }
};



int main() {
    initializer_list<int> in={0,1,1,2,3,4,5,2,3,6,7,8,9};
    CacheLRU cache(in);
    cache.work();

    return 0;
}
