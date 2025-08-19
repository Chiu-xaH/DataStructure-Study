#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    int* data;
    int size;
} MinHeap;

MinHeap* Create(int capacity) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h->data = (int*)malloc(sizeof(int) * capacity);
    h->size = 0;
    return h;
}

void Swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// 上浮
void Push(MinHeap* h, int val) {
    // 扩容
    h->size++;
    int i = h->size - 1;
    // 赋值给最后一个
    h->data[i] = val;
    // 调整堆
    while(i > 0) {
        int parent = (i - 1) / 2;
        if(h->data[i] >= h->data[parent]) {
            break;
        }
        // 交换当前节点和父节点
        Swap(&h->data[i], &h->data[parent]);
        // 更新当前节点为父节点
        i = parent;
    }
}

// 下沉
int Pop(MinHeap* h) {
    if(h->size == 0) return INT_MAX;
    int top = h->data[0];
    // 缩容
    h->size--;
    h->data[0] = h->data[h->size];
    int i = 0;
    while(2*i + 1 < h->size) {
        int l = 2*i + 1, r = 2*i + 2;
        int minChild = l;

        if(r < h->size && h->data[r] < h->data[l]) {
            minChild = r;
        }
        if(h->data[i] <= h->data[minChild]) {
            break;
        }
        Swap(&h->data[i], &h->data[minChild]);
        // 更新当前节点为最小子节点
        i = minChild;
    }
    return top;
}

// 取堆顶
int GetTop(MinHeap* h) {
    if(h->size == 0) return INT_MAX;
    return h->data[0];
}

// 力扣 2462
long totalCost(int* costs, int n, int k, int candidates) {
    int leftIndex = 0, rightIndex = n - 1;
    MinHeap* leftHeap = Create(n);
    MinHeap* rightHeap = Create(n);
    long result = 0;

    for(int i = 0; i < candidates && leftIndex <= rightIndex; i++)
        Push(leftHeap, costs[leftIndex++]);
    for(int i = 0; i < candidates && leftIndex <= rightIndex; i++)
        Push(rightHeap, costs[rightIndex--]);

    for(int i = 0; i < k; i++) {
        int leftMin = GetTop(leftHeap);
        int rightMin = GetTop(rightHeap);

        if(leftMin <= rightMin) {
            result += Pop(leftHeap);
            if(leftIndex <= rightIndex) Push(leftHeap, costs[leftIndex++]);
        } else {
            result += Pop(rightHeap);
            if(leftIndex <= rightIndex) Push(rightHeap, costs[rightIndex--]);
        }
    }

    free(leftHeap->data);
    free(leftHeap);
    free(rightHeap->data);
    free(rightHeap);

    return result;
}

// 测试
int main() {
    int costs[] = {17,12,10,2,7,2,11,20,8};
    int n = sizeof(costs)/sizeof(costs[0]);
    int k = 3, candidates = 4;
    printf("%ld\n", totalCost(costs, n, k, candidates));
    return 0;
}
