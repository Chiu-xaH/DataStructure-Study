/**
 * Note: The returned array must be malloced, assume caller calls free().
 */

/*
给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

注意：未在等式列表中出现的变量是未定义的，因此无法确定它们的答案。

输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
注意：x 是未定义的 => -1.0
*/
 /* 思路 
构建有向图，若a/b=value，构建边：a-value->b，b-value的倒数->a
若求a->c，则从a进行DFS，并累乘边的value，直到到达c，无法到达则未-1
 */ 
#define MAX 100
typedef struct EdgeNode {
    struct EdgeNode *next;
    double weight;
    int adjvex;
} EdgeNode;

typedef struct PointNode {
    EdgeNode *firstEdge;
    char data[16];  // 假设节点名最长不超过15字符
} PointNode, AdjList[MAX];

typedef struct {
    AdjList adjList;
    int numNodes;
} GraphAd;

// 查找节点名是否已经存在，存在返回索引，不存在就新增
int getNodeIndex(GraphAd* G, const char* name) {
    for (int i = 0; i < G->numNodes; i++) {
        if (strcmp(G->adjList[i].data, name) == 0)
            return i;
    }
    // 不存在就新增
    int idx = G->numNodes;
    strcpy(G->adjList[idx].data, name);
    G->adjList[idx].firstEdge = NULL;
    G->numNodes++;
    return idx;
}

int findNodeIndex(GraphAd G, const char* name) {
    for(int i=0;i<G.numNodes;i++)
        if(strcmp(G.adjList[i].data,name)==0) return i;
    return -1; // 不存在
}

// 创建有向图
void Create(GraphAd* G, char*** equations, int equationsSize, double* values) {
    G->numNodes = 0;
    for (int i = 0; i < equationsSize; i++) {
        char* start = equations[i][0];
        char* end = equations[i][1];
        double weight = values[i];

        int ii = getNodeIndex(G, start);
        int jj = getNodeIndex(G, end);

        // ii -> jj
        EdgeNode* E = (EdgeNode*)malloc(sizeof(EdgeNode));
        E->weight = weight;
        E->adjvex = jj;
        E->next = G->adjList[ii].firstEdge;
        G->adjList[ii].firstEdge = E;

        // jj -> ii
        E = (EdgeNode*)malloc(sizeof(EdgeNode));
        E->weight = 1.0 / weight;
        E->adjvex = ii;
        E->next = G->adjList[jj].firstEdge;
        G->adjList[jj].firstEdge = E;
    }
}

bool dfs(GraphAd G,int i,int end,bool* visited,double* r) {
    if(i == end) return true; 
    visited[i] = true;
    EdgeNode* edge = G.adjList[i].firstEdge;
    while(edge != NULL) {
        int node = edge->adjvex;
        if(visited[node] == false) {
            double tmp = *r * edge->weight;
            if(dfs(G, node, end, visited, &tmp)) {
                *r = tmp;
                return true;
            }
        }
        edge = edge->next;
    }
    return false;
}


double* calcEquation(char*** equations, int equationsSize, int* equationsColSize, double* values, int valuesSize, char*** queries, int queriesSize, int* queriesColSize, int* returnSize) {
    *returnSize = queriesSize;
    GraphAd G;
    Create(&G,equations,equationsSize,values);
    double* result = (double*)malloc(sizeof(double) * queriesSize);

    int index = 0;
    bool visited[G.numNodes];
    for(int i = 0;i < queriesSize;i++) {
        char* start = queries[i][0];
        char* end = queries[i][1];

        int ii = findNodeIndex(G, start);
        int jj = findNodeIndex(G, end);
        if(ii == -1 || jj == -1) {
            result[index++] = -1.0;
            continue;
        }

        // 初始化DFS Visited
        for(int i = 0;i < G.numNodes;i++) {
            visited[i] = false;
        }
        // 开始DFS 同时累乘 若无法到达终点，则为-1
        double r = 1.0;
        // 同时扩容result
        if(dfs(G, ii, jj, visited, &r) == false) {
            result[index++] = -1.0;
        } else {
            result[index++] = r;
        }
    }
    return result;
}