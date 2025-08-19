#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Stack.h"
//#include "Queue.h"

#define OK 1
#define ERROR 0
#define MAXSIZES 100 

typedef int Status;
typedef char ElemTypes;

typedef struct TreeNode {
    ElemTypes data;
    struct TreeNode *leftChild,*rightChild;
} TreeNode,*Tree;

Status Init(Tree *T) {
    (*T) = NULL;
    return OK;
}

/* 用于构造二叉树********************************** */
int treeIndex=1;
typedef char String[24]; /*  0号单元存放串的长度 */
String str;
//字符char转化为串String
Status StrAssign(String T,char *chars) { 
	if(strlen(chars)>MAXSIZES) {
        return ERROR;
    } else {
		T[0] = strlen(chars);
		for(int i = 1;i <= T[0]; i++)
			T[i]= *(chars+i-1);
		return OK;
	}
}

Status GetString(String S) {
    int length = S[0];
    for(int i = 1;i <= length;i++) {
        printf("%c",S[i]);
    }
    return OK;
}
//构造二叉树，按顺序读取串，遇到#为空，按前序遍历的顺序放置树
void CreateTree(Tree *T) { 
	ElemTypes ch;
    //从1开始，每次递归递增，移动到下一个元素
	ch=str[treeIndex++];

	if(ch=='#') {
        *T=NULL;
    } else {
		*T= (Tree)malloc(sizeof(TreeNode));
		if(!*T) exit(0);
        //采用前序遍历的方法构造
		(*T)->data=ch; /* 生成根结点 */
        //构造子树
		CreateTree(&(*T)->leftChild); 
		CreateTree(&(*T)->rightChild); 
	}
}

/* ************************************************ */

Status GetHead(Tree T) {
    if(T != NULL) {
        printf("%c",T->data);
        GetHead(T->leftChild);
        GetHead(T->rightChild);
        return OK;
    } else return ERROR;
}

Status GetMiddle(Tree T) {
    if(T != NULL) {
        GetMiddle(T->leftChild);
        printf("%c",T->data);
        GetMiddle(T->rightChild);
        return OK;
    } else return ERROR;
}

Status GetTail(Tree T) {
    if(T != NULL) {
        GetTail(T->leftChild);
        GetTail(T->rightChild);
        printf("%c",T->data);
        return OK;
    } else return ERROR;
}

int NodeCount(Tree T) {
    if(T != NULL) {
        return NodeCount(T->leftChild) + NodeCount(T->rightChild) +1;
    } else return 0;
}

/*利用现有的遍历，改造函数*/
//设计算法输出二叉树的所有叶子结点的值
Status GetLeaf(Tree T) {
    if(T != NULL) {
        if(T->leftChild == NULL && T->rightChild == NULL) {
            printf("%c",T->data);
        } 
        GetLeaf(T->leftChild);
        GetLeaf(T->rightChild);
        return OK;
    } else return ERROR;
}
int leafCount = 0;
Status LeafCount(Tree T) {
    if(T != NULL) {
        if(T->leftChild == NULL && T->rightChild == NULL) {
            leafCount++;
        } 
        LeafCount(T->leftChild);
        LeafCount(T->rightChild);
        return OK;
    } else return ERROR;
}
//设计算法输出从根结点到每个叶子结点的路径（利用栈）
Status GetLeafRoad(Tree T,Stack *S) {
    if (T == NULL) return ERROR;

    Push(S, T->data);  // 先压栈

    if (T->leftChild == NULL && T->rightChild == NULL) {
        GetStack(S);   // 到叶子节点时输出路径
    } else {
        GetLeafRoad(T->leftChild, S);
        GetLeafRoad(T->rightChild, S);
    }

    ElemTypes E;
    Pop(S,&E);  // 回溯时弹栈，撤销当前路径
    return OK;
}
//求高度
/*
（1）若T为空，则高度为0，遍历结束；
    （2）否则，设二叉树的形态如右图所示。
  a、假设左右子树能分别求出高度为hl、hr，
        则整个二叉树的高度为：
        max(hl, hr) + 1;
  b、对于左右子树高度的求解，可按照与整个二叉树相同的方式进行  （递归调用）
*/
int max(int m,int n) {
    if(m >= n) {
        return m;
    } else {
        return n;
    }
}

int Height(Tree T) {
    if(T != NULL) {
        return max(Height(T->leftChild),Height(T->rightChild)) + 1;
    } else return 0;
}

Status GetLastRight(Tree T) {
    Tree list = T;
    while(list->rightChild != NULL) {
        list = list->rightChild;
    }
    printf("The last right node is %c",list->data);
    return OK;
}

Status GetLastLeft(Tree T) {
    Tree list = T;
    while(list->leftChild != NULL) {
        list = list->leftChild;
    }
    printf("The last left node is %c",list->data);
    return OK;
}
//设计一个非递归算法以输出二叉树中先序序列中最后一个结点的值
Status GetLast(Tree T) {
    Tree list = T;
    while(list->leftChild != NULL || list->rightChild != NULL) {
        if(list->rightChild == NULL) {
            list = list->leftChild;
        } else {
            list = list->rightChild;
        } 
    }
    printf("The last node is %c",list->data);
    return OK;
}
//设计算法链表二叉树转换为顺序二叉树
void Transfer(Tree T,int i,char *S) {
    if(T != NULL) {
        S[i] = T->data;
        Transfer(T->leftChild,2*i + 1,S);
        Transfer(T->rightChild,2*i + 2,S);
    }
}
void PrintTransfer(Tree T) {
    // 计算顺序存储需要的容量 根据二叉树性质和高度
    int count = 0;
    for(int i = 0;i < Height(T);i++) {
        count += pow(2,i);
    }
    // 初始化顺序存储数组
    char A[count+1];
    for(int i = 0;i <= count;i++) {
        A[i] = '#';
    }
    // 转换
    Transfer(T,0,A);
    // 打印
    for(int i = 0;i <= count;i++) {
        printf("%c",A[i]);
    }
}
//设计算法顺序二叉树转化为链表二叉树
void Transfer2(Tree *T,int i,char *S) {
    if(S[i] == '\0' || S[i] == '^') {
        return;
    }
    if (*T == NULL) {
        *T = (Tree)malloc(sizeof(TreeNode));
        (*T)->leftChild = (*T)->rightChild = NULL;
    }

    (*T)->data = S[i];

    if (S[2*i + 1] != '\0' && S[2*i + 1] != '^') {
        Transfer2(&(*T)->leftChild, 2*i + 1, S);
    }
    if (S[2*i + 2] != '\0' && S[2*i + 2] != '^') {
        Transfer2(&(*T)->rightChild, 2*i + 2, S);
    }
}
//层序遍历
typedef struct {
    Tree data[MAXSIZE];
    int front;
    int rear;
} Queue;

Status InitQueue(Queue *Q) {
    Q->front = 0;////<--
    Q->rear = 0;//////-->
    return OK;
}

Status Add(Queue *Q,Tree E) {
    if ((Q->rear + 1) % MAXSIZE == Q->front) {
        return ERROR;
    }	
	Q->data[Q->rear] = E;	
    Q->rear++;		
	Q->rear = Q->rear % MAXSIZE;
    return OK;
}

Status DeAdd(Queue *Q,Tree *E) {
    if(Q->front == Q->rear) {
        return ERROR;
    }
    *E = Q->data[Q->front];
    Q->front++;
    Q->front = Q->front % MAXSIZE;
    return OK;
}
//通用公式
int Length(Queue Q) {
    return (Q.rear-Q.front+MAXSIZE)%MAXSIZE;
}
//队列满-通用
Status isFull(Queue Q) {
    if((Q.rear+1)%MAXSIZE == Q.front) {
        return OK;
    } else {
        return ERROR;
    }
}
//队列空
Status isEmpty(Queue Q) {
    if(Q.front == Q.rear) {
        return OK;
    } else {
        return ERROR;
    }
}
//层序遍历
void GetLayer(Tree T) {
    if (T == NULL) {
        return;
    }
    Queue Q;
    InitQueue(&Q);
    Tree tree = T;
    Add(&Q,tree);
    while(!isEmpty(Q)) {
        Tree E;
        DeAdd(&Q,&E);
        printf("%c ",E->data);
        if(E->leftChild) {
            Add(&Q,E->leftChild);
        }
        if(E->rightChild) {
            Add(&Q,E->rightChild);
        }
    }
}
//得到层数&重要算法!
int GetLayNum(Tree T,Elemtype data) {
    Queue Q;
    InitQueue(&Q);
    int count = 1;
    int nodeCount = 0,nodeNextCount = 0;

    Tree root = T;
    Add(&Q,root);
    nodeCount = 1;
    if(root == NULL) {
        return -1;
    }
    if(root->data == data) {
        return 1;
    }
    
    while(!isEmpty(Q)) {
        TreeNode E;
        DeAdd(&Q,&E);
        nodeCount--;
        if(E.data == data) {
            return count;
        }
        if(E.leftChild) {
            Add(&Q,E.leftChild);
            nodeNextCount++;
        }
        if(E.rightChild) {
            Add(&Q,E.rightChild);
            nodeNextCount++;
        }
        //准备进入下一层，初始化数据
        if(nodeCount == 0) {
            count++;
            nodeCount = nodeNextCount;
            nodeNextCount = 0;
        }
    }
    return -1;
}
//根据编号找到结点
Status Find(Tree T,int i,Tree *node) {
    Queue Q;
    InitQueue(&Q);
    if (T == NULL) {
        return ERROR;
    }

    Tree tree = T;
    int count = 0;
    Add(&Q,tree);
    while(count != i && !isEmpty(Q)) {
        Tree E;
        DeAdd(&Q,&E);
        count++;
        if(count == i) {
            *node = E;
            return OK;
        }
        if(E->leftChild) {
            Add(&Q,E->leftChild);
        }
        if(E->rightChild) {
            Add(&Q,E->rightChild);
        }
    }
    return ERROR;
}

char path2[MAXSIZE];    
//输出结点根节点到vi结点的路径
void PrintPath(char path[], int pathLength) {
    
    for (int i = 0; i < pathLength; i++) {
        path2[i] = path[i];
    }
    printf("\n");
}
int pathSize = 0;
void GetLoad(Tree T,Tree node,int pathLength,char path[]) {
    if(T == NULL) {
        return;
    }
    
    path[pathLength] = T->data;
    pathLength++;
    

    if (T == node) {
        PrintPath(path,pathLength);
        pathSize = pathLength;
    } else {
        // 否则，继续递归遍历左右子树
        GetLoad(T->leftChild,node,pathLength,path);
        GetLoad(T->rightChild,node,pathLength,path);
    }
}

void PrintPaths(Tree T,int i) {
    char path[MAXSIZE];
    Tree node = NULL;
    Find(T,i,&node);
    if (node == NULL) {
        printf("Node not found.\n");
        return;
    }
    GetLoad(T,node,0,path);
}

//设计算法输出结点到根节点路径所有结点值
int pathLen = 0;
void Print(Elemtype paths[]) {
    for(int i = 0;i < pathLen;i++) {
        printf("%c ",paths[i]);
    }
}

Status GetLoadUpdate(Tree node,Tree T,Elemtype paths[]) {
    if(T == NULL) return ERROR;

    paths[pathLen] = T->data;
    pathLen++;

    if(T == node) {
        Print(paths);
        return OK;
    } else {
        Status status = 
        GetLoadUpdate(node,T->leftChild,paths) ||
        GetLoadUpdate(node,T->rightChild,paths);

        if(!status) pathLen--;
        return ERROR;
    }
}

void Load(Tree T,int i) {
    Tree node = NULL;
    Find(T,i,&node);
    Elemtype paths[MAXSIZE];
    GetLoadUpdate(node,T,paths);
}


Status MiddleWithNum(Tree T, int *num) {
    if (T != NULL) {
        MiddleWithNum(T->leftChild, num);
        printf("(%c,%d) ", T->data, (*num)++);
        MiddleWithNum(T->rightChild, num);
        return OK;
    } else return ERROR;
}

Status GetMiddleWithNum(Tree T) {
    int num = 1;
    return MiddleWithNum(T, &num);
}

Tree SearchPublicParent(Tree T, Tree I, Tree J) {
    if (T == NULL || T == I || T == J) {
        return T;
    }

    Tree left = SearchPublicParent(T->leftChild, I, J);
    Tree right = SearchPublicParent(T->rightChild, I, J);
    // 目标在左子树和右子树，第一次发现此点即为最近公共祖先
    if (left && right) {
        return T;
    }
    // 否则返回非空的子树，两个目标都在那边
    return left ? left : right;
}

//设计算法以求解编号为i和j的两个结点的最近的公共祖先结点
void PublicParent(Tree T,int i,int j) {
    Tree node1, node2;
    if (Find(T, i, &node1) == ERROR || Find(T, j, &node2) == ERROR) {
        return;
    }

    Tree ancestor = SearchPublicParent(T, node1, node2);
    if (ancestor) {
        printf("Nearest is %c\n", ancestor->data);
    } else {
        printf("Nearest is NULL");
    }
}
// 键盘输入一个元素x，求其父节点、兄弟结点、子结点的值，不存在时给出相应提示信息。对兄弟结点和孩子结点，存在时要明确指出是左兄弟、左孩子、右兄弟或右孩子。
void PrintNodeRelation(Tree T,ElemTypes X) {
    if (T == NULL) {
        return;
    }
    Tree target = NULL;
    Tree parent = NULL;
    Tree leftBrother = NULL, rightBrother = NULL;
    Tree leftChild = NULL, rightChild = NULL;

    Queue Q;
    InitQueue(&Q);
    Tree tree = T;
    Add(&Q,tree);
    while(!isEmpty(Q)) {
        Tree E;
        DeAdd(&Q,&E);
        // 找到目标节点
        if (E->data == X) {
            target = E;
            if (E->leftChild != NULL)
                leftChild = E->leftChild;
            if (E->rightChild != NULL) 
                rightChild = E->rightChild;
        }
        if(E->leftChild) {
            Add(&Q,E->leftChild);
            if (E->leftChild->data == X) {
                parent = E;
                if (E->rightChild) rightBrother = E->rightChild;
            }
        }
        if(E->rightChild) {
            Add(&Q,E->rightChild);
            if (E->rightChild->data == X) {
                parent = E;
                if (E->leftChild) leftBrother = E->leftChild;
            }
        }
    }
    if (!target) {
        printf("Not Found Node Which Value Is %c\n", X);
        return;
    }
    if (parent) {
        printf("Parent Value Is %c\n", parent->data);
    } else {
        printf("No Parent\n");
    }
    if (leftBrother) {
        printf("Left Brother Value Is %c\n", leftBrother->data);
    }
    if (rightBrother) {
        printf("Right Brother Value Is %c\n", rightBrother->data);
    }
    if (!leftBrother && !rightBrother) {
        printf("No Brothers\n");
    }
    if (leftChild) {
        printf("Left Child Value Is %c\n", leftChild->data);
    }
    if (rightChild) {
        printf("Right Child Value Is %c\n", rightChild->data);
    }
    if (!leftChild && !rightChild) {
        printf("No Children\n");
    }
}


int count = 1;
Status GetMiddleUpdate(Tree T) {
    if(T != NULL) {
        GetMiddleUpdate(T->leftChild);
        printf(" (%c,%d) ",T->data,count);
        count++;
        GetMiddleUpdate(T->rightChild);
        return OK;
    } else return ERROR;
}
//三种遍历的非递归算法,借助栈
typedef struct {
    int top;
    Tree data[MAXSIZE];
} Stacks;

Status InitStacks(Stacks *S) {
    S->top = -1;
    return OK;
}

Status Pushs(Stacks *S,Tree *E) {
    if(S->top == MAXSIZE) {
        return ERROR;
    }
    S->top++;
    S->data[S->top] = *E;
    return OK;
}

Status Pops(Stacks *S,Tree *E) {
    if(S->top == -1) {
        return ERROR;
    }
    *E = S->data[S->top];
    S->top--;
    return OK;
}

Status AllPops(Stacks *S) {
    if(S->top == -1) {
        return ERROR;
    }
    for(int i = S->top;i > -1;i--) {
        Tree E;
        Pops(S,&E);
    }
    return OK;
}

Status GetTops(Stacks *S,Tree *E) {
    if(S->top == -1) {
        return ERROR;
    }
    *E = S->data[S->top];
    return OK;
}

void GetMiddleNot(Tree T) {
    Tree tree = T;
    Tree E;
    Stacks S;
    InitStacks(&S);
    while(tree || S.top != -1) {
        if(tree) {
            Pushs(&S,&tree);
            tree = tree->leftChild;
        } else {
            Pops(&S,&E);
            printf("%c",E->data);
            tree = E->rightChild;
        }
    }
    /*
    设置一个指针tree，初始化为根节点。
while(tree不为空 || 栈不为空)：{
	沿着当前节点的左子树向下遍历，将所有左子节点压入栈中，直到tree为空。
	从栈中弹出一个节点，访问该节点。
	将tree指向刚访问的节点的右子节点
}
    */
}

void GetHeadNot(Tree T) {
    Tree tree = T;
    Stacks S;
    InitStacks(&S);
    Pushs(&S,&tree);
    Tree E;
    while(S.top != -1) {
        Pops(&S,&E);
        printf("%c",E->data);
        if(E->rightChild) {
            Pushs(&S,&E->rightChild);
        }
        if(E->leftChild) {
            Pushs(&S,&E->leftChild);
        } 
    }
    /*
    while(栈不为空){
	 弹出栈顶元素，将该节点的值加入结果列表或输出。
 	 如果当前节点的右子节点不为空，将其压入栈中（注意：先右后左，因为栈是后进先出，保证左子树先被处理）。
	 如果当前节点的左子节点不为空，将其压入栈中。
}
    */
}

void GetTailNot(Tree T) {
    Stacks S;
    InitStacks(&S);
    Tree tree = T;
    Tree E,prv = NULL;
    while(tree || S.top != -1) {
        if(tree) {
            Pushs(&S,&tree);
            tree = tree->leftChild;
        } else {
            GetTops(&S,&E);
            if(E->rightChild == NULL || E->rightChild == prv) {
                printf("%c",E->data);
                prv = E;
                Pops(&S,&E);
            } else {
                tree = E->rightChild;
            }
        }
    }
    /*
    创建指针prv，用于记录上一个被访问的节点。
    设置指针tree，初始化为根节点。
    while(tree不为空 || 栈不为空) {
	    沿着当前节点的左子树向下遍历，将所有左子节点压入栈中，直到tree为空。
    	检查栈顶节点：
	    如果它的右子节点为空，或者右子节点已经被访问（即等于prv），那么就可以访问栈顶节点，并出栈，并将prv更新为这个节点。
	    否则，将tree指向栈顶节点的右子节点。
    }
    */
}

typedef struct {
    int count;
    int level;
} Layer;

int Question(Tree T) {
    Queue Q;
    InitQueue(&Q);
    //下标代表第几层，元素代表几个结点
    int layer[MAXSIZE];

    int level = 1;
    int nodeCount = 0,nodeNextCount = 0;

    Tree root = T;
    Add(&Q,root);
    nodeCount = 1;
    layer[level] = nodeCount;

    while(!isEmpty(Q)) {
        Tree E;
        DeAdd(&Q,&E);
        nodeCount--;
        
        if(E->leftChild) {
            Add(&Q,E->leftChild);
            nodeNextCount++;
        }
        if(E->rightChild) {
            Add(&Q,E->rightChild);
            nodeNextCount++;
        }
        //准备进入下一层，初始化数据
        if(nodeCount == 0) {
            level++;
            nodeCount = nodeNextCount;
            nodeNextCount = 0;
            layer[level] = nodeCount;
        }
    }
    level -= 1;
    //选择最多结点的一层
    int max = 1;
    for(int i = 1;i <= level;i++) {
        if(layer[i] > max) {
            max = layer[i];
        }
    }
    return max * level;
}
//已知两种遍历顺序，构造二叉树
Tree CreateUpdate(Elemtype *Head,Elemtype *Middle,int length) {
    if(length == 0) return NULL;

    Tree root = (Tree)malloc(sizeof(TreeNode));
    root->data = Head[0];

    int ptr = 0;
    while(ptr < length) {
        if(Middle[ptr] == root->data) {
            break;
        }
        ptr++;
    }
    //A BDHKE FIGJ
    //HKDBE A IFGJ
    root->leftChild = CreateUpdate(Head+1,Middle,ptr);
    root->rightChild = CreateUpdate(Head+1+ptr,Middle+ptr+1,length-1-ptr);
    return root;
}
//销毁二叉树
void Destory(Tree *T) {
    if(*T) {
        Destory(&(*T)->leftChild);
        Destory(&(*T)->rightChild);
        free(*T);
        *T = NULL;
    }
    // 字符串清空
    treeIndex = 1;
    str[0] = '\0';
}

int main() {
    Status i;  
    Tree T;
    i = Init(&T);
    /*树的结构
       A           1
      / \         / \
     B   C       2   3
    /\  /\      /\  /\
   D E F J     4 5 6 7
  /   /\      /   /\
 H   I G     8   9 10
 \           \
  K          11
    */
    StrAssign(str,"AB#CD#E###FG#HI#J####");
  //  GetString(str);
  //ABDHKECFIGJ
  //HKDBEAIFGCJ
	CreateTree(&T);
    Load(T,10);
//ABDH#K###E##CFI###G##J##
//A#BCD#E##F##G#HI###
//AB#CD#E###FG#HI#J####
//ABCD###EF###G#H#I##
//ABCD##E###F#H#IG###
//ABC##DE###FGH###I#J##
//ABC#D##E#F##GH##IJ###
//ABCDE######
//A#B#C#D#E##
//AB#CD#E####
//ABC#E##D#F##GHF##IJ####
//ABCD##E##FG###HI#JK###L##
  //  i = GetHead(T);
   // printf("\n*********************\n");
   //i = GetMiddle(T);
   // printf("\n*********************\n");
   // i = GetTail(T);
   // printf("\n*********************\n");
  //  i = GetLeaf(T);
  //  printf("\n*********************\n");
  //  Stack S;
  //  i = InitStack(&S);
  //  i = GetLeafRoad(T,&S); 
 // printf("Height = %d",Height(T));
 // int count = NodeCount(T);
  //printf("\nNode Count = %d",count);
 // printf("\n*********************\n");
//  i = LeafCount(T);
 // printf("Leaf count is %d",leafCount);
 // printf("\n*********************\n");
 // i = GetLast(T);
 // int count = 0;
 // for(int i = 0;i < Height(T);i++) {
 //   count += pow(2,i);
 // }
 // char A[count+1];
  //for(int i = 0;i <= count;i++) {
    // A[i] = 0;
 // }
 // Transfer(T,1,A);
 // for(int i = 1;i <= count;i++) {
   // printf("%c",A[i]);
 // }
    //GetLayer(T);
    //char ele = 'K';
    //printf("%c is in level %d",ele,GetLayNum(T,ele));
    //PublicParent(T,1,2);
    //Tree tree = NULL;
    //Find(T,5,&tree);
    //printf("%c",tree->data);
   // int pathLength = 0;

   // i = GetLoad(T,5,&pathLength);
   // printf("%d",i);
   // count = 1;
  //  GetMiddleUpdate(T);
  //PrintPaths(T,9);
  //  for(int i = 0;i < pathLength;i++) {
    //    printf("%c ",path[i]);
    //}
 //  GetHeadNot(T);
 //  printf("\n*********************\n");
   //GetMiddleNot(T);
  // GetTailNot(T);
  //char S[MAXSIZE];
 // for(int j = 0;j < MAXSIZE;j++) {
 ///   S[j] = '^';
 // }
  //Transfer(T,0,S);

 // Transfer2(&T,0,S);
  //i = GetHead(T);
 // PrintAllPaths(T,3);
   // char res = PublicParent(T,9,7);
    //printf("\nNearest Public is %c",res);
   // Question(T);
   //Destory(&T);
 //  if(T == NULL) printf("\nNULL");
  // else printf("\nNO");
  //ABDHKECFIGJ
  //HKDBEAIFGCJ
  //Elemtype A[] = {'A','B','D','H','K','E','C','F','I','G','J'};
  //Elemtype B[] = {'H','K','D','B','E','A','I','F','G','C','J'};
  //Tree Ts = CreateUpdate(A,B,11);
  //GetTail(Ts);
 // GetHeadNot(T);
}