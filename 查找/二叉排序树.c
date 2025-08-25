#include <stdio.h>
#include <stdlib.h>
#include "Queue.h"
#include <limits.h>
#define OK 1
#define ERROR 0
#define MAXSIZE 100
#define SIZE 9

typedef int Status;
typedef int ElemType;

typedef struct TreeNode{
    ElemType data;
    struct TreeNode *leftChild,*rightChild;
} TreeNode,*Tree;

Status Init(Tree *T) {
    (*T) = NULL;
    return OK;
}

Status SearchRecursion(Tree T,ElemType key) {
    if(T == NULL) {
        printf("NO\n");
        return ERROR;
    }
    if(key == T->data) {
        printf("YES\n");
        return OK;
    } else if(key < T->data) {
        return SearchRecursion(T->leftChild,key);
    } else {
        return SearchRecursion(T->rightChild,key);
    }
}

Status Search(Tree T,ElemType key) {
    Tree tree = T;
    while(tree != NULL) {
        if(key < tree->data) {
            tree = tree->leftChild;
        } else if(key > tree->data) {
            tree = tree->rightChild;
        } else {
            //printf("YES\n");
            return OK;
        }
    }
    //printf("NO\n");
    return ERROR;
}
//插入一个值
Status Insert(Tree *T,ElemType key) {
    if(Search(*T,key) == ERROR) {
        Tree new = (Tree)malloc(sizeof(TreeNode));
        new->data = key;
        new->leftChild = NULL;
        new->rightChild = NULL;
        
        if(*T == NULL) {
            *T = new;
        } else {
            //找到叶子
            Tree tree = *T;
            while(tree != NULL) {
                if(key < tree->data) {
                    if(tree->leftChild == NULL) break;
                    tree = tree->leftChild;
                } else if(key > tree->data) {
                    if(tree->rightChild == NULL) break;
                    tree = tree->rightChild;
                }
            }
            if(key < tree->data) {
                tree->leftChild = new;
            } else if(key > tree->data) {
                tree->rightChild = new;
            }
        }
        return OK;
    } else {
        return ERROR;
    }
}

Status Delete(Tree *T,ElemType key) {
    Tree tree = *T;
    Tree parent = NULL;
    while(tree != NULL && key != tree->data) {
        parent = tree;
        if(key < tree->data) {
            tree = tree->leftChild;
        } else if(key > tree->data) {
            tree = tree->rightChild;
        }
    }

    if(tree == NULL) {
        return ERROR;
    }

    //仅有左子树的结点或叶子结点
    if(tree->rightChild == NULL) {
        Tree old = tree;
        tree = tree->leftChild;
        free(old);
    } else
    //仅有右子树的结点
    if(tree->leftChild == NULL) {
        Tree old = tree;
        tree = tree->rightChild;
        free(old);
    } else
    //左右子树均有的结点
    {
        Tree successor = tree->rightChild;
        Tree successorParent = tree;
        // 方法1寻找右子树最小节点
        //方法2 寻找左子树最大结点
        while (successor->leftChild != NULL) {
            successorParent = successor;
            successor = successor->leftChild;
        }
        // 替换节点数据
        tree->data = successor->data;
        // 调整指针删除successor节点
        if (successorParent->leftChild == successor) {
            //重接左子树
            successorParent->leftChild = successor->rightChild;
        } else {
            //重接右子树
            successorParent->rightChild = successor->rightChild;
        }
        free(successor);
    }
}

Status GetMiddle(Tree T) {
    if(T != NULL) {
        GetMiddle(T->leftChild);
        printf("%d ",T->data);
        GetMiddle(T->rightChild);
        return OK;
    } else return ERROR;
}

ElemType GetMin(Tree T) {
    Tree tree = T;
    while(T->leftChild != NULL) {
        T = T->leftChild;
    }
    return T->data;
}

ElemType GetMax(Tree T) {
    Tree tree = T;
    while(T->rightChild != NULL) {
        T = T->rightChild;
    }
    return T->data;
}

typedef struct {
    Tree data[MAXSIZE];
    int front;
    int rear;
} Queue;

Status InitQueue(Queue *Q) {
    Q->front = 0;
    Q->rear = 0;
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

int Length(Queue *Q) {
    if(Q->front == Q->rear) {
        return 0;
    }
    int count = 0;
    int i = Q->front;
    while(i != Q->rear) {
        count++;
        i++;
        i = i % MAXSIZE;
    }
    return count;
}
//设计算法求出平均查找长度
int AverangeLength(Tree T) {
    Tree tree = T;
    Queue Q;
    InitQueue(&Q);
    Add(&Q,tree);
    int level = 0;
    int total;
    while(Length(&Q) != 0) {
        level++;
        int nodeCount = Length(&Q);
        total += level * nodeCount;
        while(nodeCount > 0) {
            Tree E;
            DeAdd(&Q,&E);
            if(E->leftChild != NULL) {
                Add(Q,E->leftChild);
            }
            if(E->rightChild != NULL) {
                Add(&Q,E->rightChild);
            }
            nodeCount--;
        } 
    }  
    return total/SIZE;  
}
//给定二叉树，各结点值不同，设计算法判断是否为二叉排序树
void TransferArray(Tree T,int array[],int *place) {
    if(T != NULL) {
        TransferArray(T->leftChild,array,place);
        array[*place] = T->data;
        (*place)++;
        TransferArray(T->rightChild,array,place);
    }
}

Status isBinaryTree(Tree T) {
    //二叉排序树中序遍历后，一定是有顺序的序列，判定序列是否有顺序即可
    //二叉排序树的结点左孩子<根节点<右孩子，判定大小关系确定是否为平衡二叉树
    int array[SIZE];
    int place = 0;
    TransferArray(T,array,&place);
    int ptr = 0;
    while(ptr+1 < SIZE) {
        if(array[ptr] > array[ptr+1]) {
            return ERROR;
        } 
        ptr++;
    }
    return OK;
}


Status helper(Tree T, long min, long max) {
    if (T == NULL) return OK;
    if (T->data <= min || T->data >= max) return ERROR;
    return helper(T->leftChild, min, T->data) && helper(T->rightChild, T->data, max);
}

Status isBST(struct TreeNode* root) {
    return helper(root, LONG_MIN, LONG_MAX);
}

int main() {
    ElemType arr[9] = { 0,1,2,3,4,5,6,7,8 };
    Tree T;
    Status I;
    I = Init(&T);
    for(int i = 0;i < SIZE;i++) {
        I = Insert(&T,arr[i]);
    }

   // int key;
  //  printf("Input you want to serch\n");
   // scanf("%d",&key);

   // I = SearchRecursion(T,key);
   //I = GetMiddle(T);
   //printf("\n");
   //printf("\nmax is %d",GetMax(T));
   printf("length is %d",AverangeLength(T));
   //I = isBinaryTree(T);
   //printf("{%d}",I);
}