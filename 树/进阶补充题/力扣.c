#include <stdio.h>
#include <stdlib.h>

const int true = 1,false = 0;
typedef int bool;

struct TreeNode {
      int val;
      struct TreeNode *left;
      struct TreeNode *right;
};

//最小深度：从根节点到最近叶子节点的最短路径上的节点数量。
int min(int m,int n) {
    return m < n ? m : n;
}

int minDepth(struct TreeNode* root) {
    if(root==NULL) return 0;
    if(root->left == NULL) return 1 + minDepth(root->right);
    if(root->right == NULL) return 1 + minDepth(root->left);
    // 如果左右子树都不为空，返回左右子树最小深度 + 1
    return 1 + min(minDepth(root->left), minDepth(root->right));
}

//翻转二叉树
void PreOrder(struct TreeNode * root) {
    if(root) {
        struct TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;
        PreOrder(root->left);
        PreOrder(root->right);
    }
}

struct TreeNode* invertTree(struct TreeNode* root) {
    PreOrder(root);
    return root;
}


//判断是否是对称二叉树
bool isMirror(struct TreeNode* left,struct TreeNode* right) {
    if(left == NULL && right == NULL) {
        return true;
    } else if(left != NULL && right == NULL) {
        return false;
    } else if(left == NULL && right != NULL) {
        return false;
    } else  {
        //核心思想：细化成最小子问题
        //左子树的左子树对称为右子树的右子树，左子树的右子树对称为右子树的左子树
        bool structEqual = isMirror(left->left,right->right) && isMirror(right->left,left->right);
        bool valEqual = (left->val == right->val);
        return valEqual && structEqual;
    }
}

bool isSymmetric(struct TreeNode* root) {
    if(root) {
       return isMirror(root->left,root->right);
    }
    return true;
}

//判断是否是相同树
bool isSameTree(struct TreeNode* p, struct TreeNode* q) {
    if(p && q) {
        bool valEqual = (p->val == q->val);
        bool structEqual = isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
        return valEqual && structEqual;
    } else if(!p && !q) {
        return true;
    } else {
        return false;
    }
}

//判断是否为AVL树
int max(int a,int b) {
    return a > b ? a : b;
}

int Height(struct TreeNode* root) {
    if(root) {
        return 1 + max(Height(root->left),Height(root->right));
    } else return 0;
}

int BH(struct TreeNode* root) {
    int left = Height(root->left);
    int right = Height(root->right);
    return left - right;
}

bool isBalanced(struct TreeNode* root) {
    if (!root) return true;
    int bh = BH(root);
    bool structEqual = isBalanced(root->left) && isBalanced(root->right);
    return (fabs(bh) <= 1) && structEqual;
}

//判断是否是堂兄弟结点（父节点不同，但深度相同）
 //思路：先找到x和y结点的父节点，父节点不同但深度相同则为堂兄弟
struct TreeNode* PreOrder(struct TreeNode* root,struct TreeNode* parent,int a) {
    if(root) {
        if(root->val == a) {
            return parent;
        }
        if(PreOrder(root->left,root,a)) return PreOrder(root->left,root,a);
        return PreOrder(root->right,root,a);
    }
    return NULL;
}
int Height(struct TreeNode* root,int a) {
    if (root == NULL) return -1;
    if (root->val == a) return 0;
    int leftHeight = Height(root->left,a);
    if (leftHeight != -1) return leftHeight + 1;

    int rightHeight = Height(root->right, a);
    if (rightHeight != -1) return rightHeight + 1;

    return -1;
}

bool isCousins(struct TreeNode* root, int x, int y) {
    struct TreeNode* parentX = PreOrder(root,NULL,x);
    struct TreeNode* parentY = PreOrder(root,NULL,y);
    if(parentX != parentY && Height(root,x) == Height(root,y)) {
        return true;
    } else {
        return false;
    }
}
//计算左叶子总和数
void PreOrder(struct TreeNode* root,int *total) {
    if(root) {
        if(root->left && root->left->left == NULL && root->left->right == NULL) {
            *total += root->left->val;
        }
        PreOrder(root->left,total);
        PreOrder(root->right,total);
    }
    
}
 
int sumOfLeftLeaves(struct TreeNode* root) {
    int total = 0;
    PreOrder(root,&total);
    return total;
}

 //二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
//两节点之间路径的 长度 由它们之间边数表示。

 //思路：直径=根左子树高度+根右子树高度
 //子树高度用递归或者遍历思想
 int max(int a,int b) {
    return a > b ? a : b;
 }
int Height(struct TreeNode* root,int *dis) {
    if(root) {
        int left = Height(root->left,dis);
        int right = Height(root->right,dis);
        *dis = max(*dis,left+right);
        return 1 + max(left,right);
    } else return 0;
}

int diameterOfBinaryTree(struct TreeNode* root) {
    int dis = 0;
    Height(root,&dis);
    return dis;
}
//判断sub是否是root子树
bool isSameTree(struct TreeNode* s, struct TreeNode* t) {
    if (!s && !t) return true;
    if (!s || !t) return false;
    return (s->val == t->val) && isSameTree(s->left, t->left) && isSameTree(s->right, t->right);
}

bool isSubtree(struct TreeNode* root, struct TreeNode* subRoot) {
    if (!root) return false;
    if (isSameTree(root, subRoot)) return true;
    return isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
}
//叶子相似的树

//请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。
//举个例子，如上图所示，给定一棵叶值序列为 (6, 7, 4, 9, 8) 的树。
//如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。
//如果给定的两个根结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。
void PreOrder(struct TreeNode* tree,int* leafList,int *size) {
    if(tree) {
        if(tree->left == NULL && tree->right == NULL) {
            leafList[*size] = tree->val;
            (*size)++;
        }
        PreOrder(tree->left,leafList,size);
        PreOrder(tree->right,leafList,size);
    }
}

bool leafSimilar(struct TreeNode* root1, struct TreeNode* root2) {
    int leafList1[199];
    int leafList2[199];
    int size1 = 0,size2 = 0;
    PreOrder(root1,leafList1,&size1);
    PreOrder(root2,leafList2,&size2);
    if(size1 == size2) {
        for(int i = 0;i < size1;i++) {
            if(leafList1[i] != leafList2[i]) {
                return false;
            }
        }
        return true;
    } else {
        return false;
    }
}

 
//单值二叉树：判断所有结点是否都为一个值
bool PreOrder(struct TreeNode* tree,int value) {
    if(tree != NULL) {
        if(tree->val != value) return false;
        return PreOrder(tree->left,value) && PreOrder(tree->right,value);
    }
    return true;
}

bool isUnivalTree(struct TreeNode* root) {
    int value = root->val;
    return PreOrder(root,value);
}
//二叉树额坡度
/*
一个树的 节点的坡度 定义即为，该节点左子树的节点之和和右子树节点之和的 差的绝对值 。如果没有左子树的话，左子树的节点之和为 0 ；没有右子树的话也是一样。空结点的坡度是 0 。

整个树 的坡度就是其所有节点的坡度之和。
示例 ：
    4
   / \
  2  9
 /\   \
3 5    7
输出：15
解释：
节点 3 的坡度：|0-0| = 0（没有子节点）
节点 5 的坡度：|0-0| = 0（没有子节点）
节点 7 的坡度：|0-0| = 0（没有子节点）
节点 2 的坡度：|3-5| = 2（左子树就是左子节点，所以和是 3 ；右子树就是右子节点，所以和是 5 ）
节点 9 的坡度：|0-7| = 7（没有左子树，所以和是 0 ；右子树正好是右子节点，所以和是 7 ）
节点 4 的坡度：|(3+5+2)-(9+7)| = |10-16| = 6（左子树值为 3、5 和 2 ，和是 10 ；右子树值为 9 和 7 ，和是 16 ）
坡度总和：0 + 0 + 0 + 2 + 7 + 6 = 15
*/

//找出BST中的众数
//统计树的结点数
void count(struct TreeNode* root,int *num) {
    if(root) {
        (*num)++;
        count(root->left,num);
        count(root->right,num);
    }
}

void inorder(struct TreeNode* root, int* prev, int* count, int* maxCount, int* result, int* returnSize) {
    if (root == NULL) return;
    
    inorder(root->left, prev, count, maxCount, result, returnSize);
    
    if (*prev == root->val) {
        (*count)++;  // 当前值与前一个值相同，计数加1
    } else {
        *count = 1;  // 否则重置计数
    }
    
    if (*count > *maxCount) {
        *maxCount = *count;
        *returnSize = 1;  // 更新最大计数，并将结果数组重置
        result[0] = root->val;
    } else if (*count == *maxCount) {
        result[*returnSize] = root->val;  // 如果当前计数等于最大计数，保存当前值
        (*returnSize)++;
    }
    
    *prev = root->val;  // 更新前一个值为当前值
    
    inorder(root->right, prev, count, maxCount, result, returnSize);
}

int* findMode(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    if (root == NULL) return NULL;
    
    int counts = 0;
    count(root, &counts); 
    
    int* result = (int*)malloc(sizeof(int) * counts);  // 动态分配空间，存储结果
    
    int prev = 0;
    int count = 0;
    int maxCount = 0;
    
    inorder(root, &prev, &count, &maxCount, result, returnSize);  // 中序遍历
    
    return result;
}

/*
给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。
更正式地说，即 root.val = min(root.left.val, root.right.val) 总成立。
给出这样的一个二叉树，你需要输出所有节点中的 第二小的值 。
如果第二小的值不存在的话，输出 -1 。
示例
    2
   / \
  2   5
     / \
    5   7
*/
int findSecondMinimumValue(struct TreeNode* root) {
    if(root == NULL) return -1;
    if(root->left && root->right) {
        int left = root->left->val == root->val ? findSecondMinimumValue(root->left) : root->left->val;
        int right = root->right->val == root->val ? findSecondMinimumValue(root->right) : root->right->val;
        if(left == -1) return right;
        if(right == -1) return left;
        return left < right ? left : right;
    } else {
        return -1;
    }
}

//树的右视图
 //思路：如果结点有右孩子 展示右孩子
 //如果结点无右孩子有左孩子 展示左孩子
 //如果结点无右孩子无左孩子（叶子） 谁也不展示
void order(struct TreeNode* root,int *arr,int *size,int height) {
    if(root == NULL) return;
    if(height == *size) {
        arr[(*size)++] = root->val;
    }
    order(root->right,arr,size,height+1);
    order(root->left,arr,size,height+1);
}
int* rightSideView(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    int* arr = (int*)malloc(sizeof(int)*100);
    order(root,arr,returnSize,0);
    return arr;
}


//找树最左下角值
 int max(int a,int b) {
    return a > b ? a : b;
 }
 int height(struct TreeNode* root) {
    if(root == NULL) return 0;
    return max(height(root->left),height(root->right))+1;
 }
 void order(struct TreeNode* root,int *num,int hei,int cur) {
    if(root ==NULL) return;
    if(hei == cur) *num = root->val;
    
    order(root->right,num,hei,cur+1);
    order(root->left,num,hei,cur+1);
 }
int findBottomLeftValue(struct TreeNode* root) {
    int num;
    order(root,&num,height(root),1);
    return num;
}

//合并二叉树（值也要合并）
struct TreeNode* mergeTrees(struct TreeNode* root1, struct TreeNode* root2) {
    if(root1 == NULL) return root2;
    if(root2 == NULL) return root1;
    root1->val += root2->val;
    root1->left = mergeTrees(root1->left,root2->left);
    root1->right = mergeTrees(root1->right,root2->right);
    return root1;
}
//二叉树剪枝 原二叉树结点要么1要么0，返回移除所有不包含1的子树的二叉树
struct TreeNode* pruneTree(struct TreeNode* root) {
    if(root == NULL) return root;
    
    root->left = pruneTree(root->left);
    root->right = pruneTree(root->right);
    //按后序遍历看，就是依次移除叶子0结点，如果用前序遍历或者中序遍历麻烦的多
    if (root->val == 0 && root->left == NULL && root->right == NULL) {
        return NULL;
    }
    return root;
}

/*
给你一棵二叉树的根节点 root ，二叉树中节点的值 互不相同 。另给你一个整数 start 。在第 0 分钟，感染 将会从值为 start 的节点开始爆发。
每分钟，如果节点满足以下全部条件，就会被感染：
->节点此前还没有感染。
->节点与一个已感染节点相邻。
返回感染整棵树需要的分钟数。
*/
//解题思路：树转化为无向图 BFS搜索即可
//需要用到三个数据结构：树（二叉链表树） 图（邻接表） 队列
//由于我自己写的代码过于冗余，就贴上其他优秀力扣用户的了
//C++
class Solution {
public:
    using IntTreeNode = TreeNode;
    int DfsTree(IntTreeNode* root, int start, int use_time, bool& visit, int& res, int& start_deep, int cur_deep)
    {
        if (!root)return 0;

        if (root->val == start)
        {
            visit = true;
            start_deep = cur_deep;
        }

        auto last_visit = visit;
        //当前节点是start的子节点或者同属于一个子树
        if (last_visit && !(root->val == start))
        {
            ++use_time;
            res = max(use_time, res);
        }

        auto left_deep = DfsTree(root->left, start, use_time, visit, res, start_deep, cur_deep+1);
        auto left_vist = visit;
        //start在左子树
        if (!last_visit && left_vist)
        {
            use_time += start_deep-cur_deep;
            res = max(use_time, res);
        }

        auto right_deep = DfsTree(root->right, start, use_time, visit, res, start_deep, cur_deep + 1);
        auto right_visit = visit;
        //start在右子树
        if (!left_vist && right_visit)
        {
            use_time += left_deep + start_deep - cur_deep;
            res = max(use_time, res);
        }
        
        //返回当前子树的深度
        return max(right_deep, left_deep) + 1;
    }

    int amountOfTime(IntTreeNode* root, int start)
    {
        int res{ 0 }, use_time{ 0 }, start_deep{ 0 };
        bool visit{ false };
        DfsTree(root, start, use_time, visit, res, start_deep,0);
        return res;
    }
};



/*

给你一棵以 root 为根的二叉树，二叉树中的交错路径定义如下：

选择二叉树中 任意 节点和一个方向（左或者右）。
如果前进方向为右，那么移动到当前节点的的右子节点，否则移动到它的左子节点。
改变前进方向：左变右或者右变左。
重复第二步和第三步，直到你在树中无法继续移动。
交错路径的长度定义为：访问过的节点数目 - 1（单个节点的路径长度为 0 ）。

请你返回给定树中最长 交错路径 的长度。
*/
int order(struct TreeNode* tree,bool goToLeft) {
    if(tree == NULL) {
        return 0;
    }
    if(goToLeft) {
        return order(tree->left,false) + 1;
    } else {
        return order(tree->right,true) + 1;
    }
}
int max(int a,int b) {
    if(a > b) {
        return a;
    } else {
        return b;
    }
}
// 得到结点的最长交错路径长度
int nodeMaxLength(struct TreeNode* tree) {
    int left = order(tree->right,true);
    int right = order(tree->left,false);
    return max(left,right);
}
void preOrder(struct TreeNode* root,int* maxValue) {
    if(root == NULL) {
        return;
    }
    if(root->left == NULL && root->right == NULL) {
        return;
    }
    int currentValue = nodeMaxLength(root);
    if(currentValue > *maxValue) {
        *maxValue = currentValue;
    }
    preOrder(root->left,maxValue);
    preOrder(root->right,maxValue);
}
// 遍历每一个非叶子结点的nodeMaxLength比较最大值
int longestZigZag(struct TreeNode* root) {
    int maxValue = 0;
    preOrder(root,&maxValue);
    return maxValue;
}