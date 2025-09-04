# 数据结构与算法
## 说明
自学数据结构的所有实践代码和学习资料，适用于期末考试或23级转专业，每个函数都可能在试卷上出现

文件夹中的进阶补充题感兴趣可看，是考后刷算法题补充的

**24级转专业考核题型、机考平台有变化，请向他们求教🫶**

## 开始学习
用编程语言实现每章的数据结构及其PPT习题、教材习题，优先推荐带指针的C

C语言的要求：
typedef、struct(结构体)、malloc与free函数(内存分配与销毁)、指针、掌握至少一种排序的写法、while,if,for等语句、enum(枚举)

以实现栈(Stack)为例：
```C
#include <stdio.h>
#include <stdbool.h> // 使用布尔(bool)值所需头文件
#define MAXSIZE 200 // 定义最大容量
// 结构体定义栈
typedef struct {
    int top; // 栈顶指针
    Elemtype data[MAXSIZE]; //数据
} Stack;
// Init为初始化，如果需要修改Stack，传入参数为指针，即Stack *S，如果不需要修改只读取Stack的数据，传入Stack S，注意观察Print与Push之间的差别，传入参数方式的不同使写法不同，S->top或S.top
void Init(Stack *S) {
    S->top = -1;
}
// 进栈
bool Push(Stack *S,int E) {
    if(S->top == MAXSIZE) return false; // 如果栈满

    S->top++; // +1
    S->data[S->top] = E; // 压入数据
    return true;
}

bool Print(Stack S) {
    if(S.top == -1) return false; // 如果栈空

    for(int i = S.top;i > -1;i--) { // 打印所有栈
        printf("%d ",S.data[i]);
    }
    printf("]\n");
    
    return true;
}

int main() {
    Stack S;
    Init(&S);
    Push(&S,1);
    Print(S);
}
```

### [应对转专业](/EXAM.md)

### 应对期末考试
动手用代码实现一遍数据结构和PPT中的问题及其习题，多刷刷题，看答案，考的都比较固定
### 主要参考资料：

**《大话数据结构》**

**《数据结构》** 胡学纲主编 安徽大学出版社 习题部分

 Github中多位学长开源的资料

 工大小题-数据结构 期末真题

### 补充
感谢以上给予我的帮助🙏得益于23年新政策，目前已成功转入至理想专业； 如有补充或纠正,欢迎帮助补充 zsh0908@outlook.com