#include <stdlib.h>
#include <stdio.h>

#define OK 1
#define ERROR 0
#define MAXSIZE 100
#define NULLKEY 0

typedef int Status;

typedef struct {
    int array[MAXSIZE];
    int length;
} List;

Status Init(List *L,int len,int arr[]) {
    L->length = len;
    for(int i = 0;i < L->length;i++) {
        L->array[i] = arr[i];
    }
    return OK;
}

void Swap(List *L,int i,int j) {
    int temp = L->array[i];
    L->array[i] = L->array[j];
    L->array[j] = temp;
}

void Get(List L) {
    for(int i = 0;i < L.length;i++) {
        printf("%d ",L.array[i]);
    }
    printf("\n");
}
//冒泡
void BubbleSort(List *L) {
    for(int i = 0;i < L->length;i++) {
        for(int j = 1;j <= L->length-1-i;j++) {
            if(L->array[j] > L->array[j+1]) {
                Swap(L,j,j+1);
            }
        }
    }
}
//快速
void QuickSortRecursive(List *L,int start,int end) {
    if(start >= end) return;
    //以开头作为枢轴
    int center = L->array[start];
    int left = start+1,right = end;
    while(left <= right) {
        while(left <= right && L->array[left] < center) {
            left++;
        }
        while(left <= right && L->array[right] > center) {
            right--;
        }
        if(left <= right) {
            //循环结束，说明有元素需要交换调整
            Swap(L,left,right);
            left++;
            right--;
        }
    }
    //枢轴回到空格位置
    Swap(L,start,right);

    QuickSortRecursive(L,start,right-1);
    QuickSortRecursive(L,right+1,end);
}

void QuickSort(List *L) {
    QuickSortRecursive(L,0,L->length-1);
}
//选择
void SelectSort(List *L) {
    for(int i = 0;i < L->length;i++) {
        int min = i;
        for(int j = i+1;j < L->length;j++) {
            if(L->array[j] < L->array[min]) {
                min = j;
            }
            if(min != i) {
                Swap(L,i,min);
            }
        }
    }
}
//插入
void InsertSort(List *L) {
    for(int i = 1;i < L->length;i++) {
        int temp = L->array[i];
        int ptr = i-1;
        //腾出空位
        while(ptr >= 0 && L->array[ptr] > temp) {
            L->array[ptr+1] = L->array[ptr];
            ptr--;
        }
        //插入
        L->array[ptr+1] = temp;
    }
}
//希尔
void ShellSort(List *L) {
    int gap = L->length/2;
    while(gap > 0) {
        for(int i = gap;i < L->length;i++) {
            int temp = L->array[i];
            int ptr = i - gap;
            while(ptr >= 0 && L->array[ptr] > temp) {
                L->array[ptr+gap] = L->array[ptr];
                ptr -= gap;
            }
            L->array[ptr+gap] = temp;
        }
        gap /= 2;
    }
}
//归并
//将SR[start...middle]与SR[middle+1...end]合并为TR[start...end]
void Merge(int SR[],int TR[],int start,int middle,int end) {
    int i = start,j = middle+1,k = start;
    while(i <= middle && j <= end) {
        if(SR[i] <= SR[j]) {
            TR[k] = SR[i];
            i++;
        } else {
            TR[k] = SR[j];
            j++;
        }
        k++;
    }
    while(i <= middle) {
        TR[k] = SR[i];
        i++;k++;
    }
    while(j <= end) {
        TR[k] = SR[j];
        j++;k++;
    }
}

void MergeSortRecursive(int SR[],int TR[],int start,int end) {
    int TR2[MAXSIZE];
    if(start == end) {
        TR[start] = SR[start];
    } else {
        int middle = (start+end) / 2;
        MergeSortRecursive(SR,TR2,start,middle);
        MergeSortRecursive(SR,TR2,middle+1,end);
        Merge(TR2,TR,start,middle,end);
    }
}

void MergeSort(List *L) {
    MergeSortRecursive(L->array,L->array,0,L->length-1);
}
//堆
void HeapAdjust(List *L,int start,int end) {
    int ptr = start,left = start*2+1,right = start*2+2;
    if(left < end && L->array[left] > L->array[ptr]) {
        ptr = left;
    }
    if(right < end && L->array[right] > L->array[ptr]) {
        ptr = right;
    }
    if(ptr != start) {
        Swap(L,start,ptr);
        HeapAdjust(L,ptr,end);
    }
}

void HeapSort(List *L) {
    for(int i = L->length / 2 - 1;i >= 0;i--) {
        HeapAdjust(L,i,L->length);
    }
    for(int i = L->length - 1;i > 0;i--) {
        Swap(L,0,i);
        HeapAdjust(L,0,i);
    }
}
