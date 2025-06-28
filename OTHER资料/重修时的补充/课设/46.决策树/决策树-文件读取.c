#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define MAX_LINE_LENGTH 256 // 文件的最大行数
#define MAX_FEATURES 64  // 最大特征数
// 文件书写格式：每行以空格或制表符分隔的整数，前面代表分类标签，用整数从1开始表示，train的每行最后一位2代表Y，1代表N，而test没有分类标签，只有特征值
/*
例如 
    // 待分类数据集
    Data testData[] = {
        {SUNNY, HOT, DRY, CALM, Nil},
        {SUNNY, HOT, DRY, WINDY, Nil},
        {RAINY, HOT, DRY, CALM, Nil},
        {SUNNY, MEDIUM, DRY, CALM, Nil},
        {SUNNY, COLD, HUMID, WINDY, Nil},
        {SUNNY, COLD, HUMID, CALM, Nil}
    };
    // 训练数据集
    Data originalData[] = {
        {SUNNY, HOT, HUMID, CALM, N},
        {SUNNY, HOT, HUMID, WINDY, N},
        {CLOUDY, HOT, HUMID, CALM, Y},
        {RAINY, MEDIUM, HUMID, CALM, Y},
        {RAINY, COLD, DRY, CALM, Y},
        {RAINY, COLD, DRY, WINDY, N},
        {CLOUDY, COLD, DRY, WINDY, Y},
        {SUNNY, MEDIUM, HUMID, CALM, N},
        {SUNNY, COLD, DRY, CALM, Y},
        {RAINY, MEDIUM, DRY, CALM, Y},
        {SUNNY, MEDIUM, DRY, WINDY, Y},
        {CLOUDY, MEDIUM, HUMID, WINDY, Y},
        {CLOUDY, HOT, DRY, CALM, Y},
        {RAINY, MEDIUM, HUMID, WINDY, N},
    };
对应
    // 待分类数据集
    1 1 2 2
    1 1 2 1
    2 1 2 2
    1 3 2 2
    1 2 1 1
    1 2 1 2
    // 训练数据集
    1 1 1 2 1
    1 1 1 1 1
    3 1 1 2 2
    2 3 1 2 2
    2 2 2 2 2
    2 2 2 1 1
    3 2 2 1 2
    1 3 1 2 1
    1 2 2 2 2
    2 3 2 2 2
    1 3 2 1 2
    3 3 1 1 2
    3 1 2 2 2
    2 3 1 1 1
*/

#define TRAINING_FILE_PATH "D:\\train.txt" // 训练数据
#define TEST_FILE_PATH "D:\\test.txt" // 要分类数据
#define TREE_IMAGE_PATH "D:\\tree.dot" // 生成可视化
 
typedef int Type;
#define N 1
#define Y 2
#define Nil 0
// 特征索引
typedef int Feature;
// 训练数据集
typedef struct Data {
    Feature* features;    // 特征数组（长度为 featureCount）
    Type classify;    // 分类标签（Y/N），测试集为 Nil
} Data;
// Log2 数学公式
double log2_(double x) {
    return log(x) / log(2); 
}
// 计算熵 熵=求和(-p * logp)
double calcEntropy(Data* data, int size) {
    // Y的数量 N的数量
    int countY = 0, countN = 0;
    for (int i = 0; i < size; ++i) {
        if (data[i].classify == Y) countY++;
        else if (data[i].classify == N) countN++;
    }
    // 计算p（概率）=数量/总数量
    double py = (double)countY / size; // Y分类的p
    double pn = (double)countN / size; // N分类的p
    double entropy = 0; 
    if (py > 0) entropy -= py * log2_(py);
    if (pn > 0) entropy -= pn * log2_(pn);
    return entropy;
}
// 获取某个特征的最大可能取值
int getFeatureMaxValue(Data* data, int size, Feature featureIndex) {
    int max = 0;
    for (int i = 0; i < size; i++) {
        if (data[i].features[featureIndex] > max) {
            max = data[i].features[featureIndex];
        }
    }
    return max;
}
// 获取指定特征值对应的子数据集
int splitData(Data* data, int size, Feature featureIndex, Feature value, Data* out, int featureCount) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        // 如果特征值匹配，则加入数组 
        // 例如featureIndex为FEATURE_WEATHER，将数据集的天气特征值赋给val，然后比较传入的特征值value是否匹配，是则加入数组
        if (data[i].features[featureIndex] == value) {
            out[count].features = malloc(sizeof(int) * featureCount); // 必须复制特征
            memcpy(out[count].features, data[i].features, sizeof(int) * featureCount);
            out[count].classify = data[i].classify;
            count++;
        }
    }
    return count;
}
// 计算信息增益(IG) IG=父节点的熵 - 求和(子节点的熵 * 子结点的占比)
double calcGain(Data* data, int size, Feature featureIndex, int featureCount) {
    // 父结点的熵
    double entropyBefore = calcEntropy(data, size);
    // 子节点的熵
    double entropyAfter = 0.0;
    // 该Feature的最大有几种情况 例如天气Feature有晴、雨、多云，则为3
    // 此值将决定求和几次 或者说有几个子节点
    int maxValue = getFeatureMaxValue(data, size, featureIndex);
    // 子节点
    Data* subset = malloc(sizeof(Data) * size);
    for (int v = 1; v <= maxValue; v++) {
        int subsetSize = splitData(data, size, featureIndex, v, subset,featureCount);
        // 子集为空，跳过
        if (subsetSize == 0) continue;
        // 子集的熵
        double subEntropy = calcEntropy(subset, subsetSize);
        // 求和 (对应上的数量/总数)*熵
        entropyAfter += ((double)subsetSize / size) * subEntropy;
        for (int i = 0; i < subsetSize; i++) {
            free(subset[i].features);
        }
    }
    free(subset);
    // IG
    return entropyBefore - entropyAfter;
}
// 判断是否为纯叶子 即所有数据都属于同一分类，无需再划分
bool isPure(Data* data, int size) {
    Type first = data[0].classify;
    for (int i = 1; i < size; ++i) {
        if (data[i].classify != first) return false;
    }
    return true;
}
// 决策树
typedef struct TreeNode {
    Feature featureIndex; // 按哪个特征划分
    Feature featureValue; // 当前节点是该特征的哪个取值（从父节点分裂过来）
    Type classify;    // 若是叶子，分类为N或Y
    int childrenCount;
    struct TreeNode** children; // 孩子数组，最大值应为特征的最大取值数getFeatureMaxValue(featureIndex)
    Feature* childFeatureValues; // 子节点对应的特征值
} TreeNode,*Tree;
// 建立决策树
TreeNode* buildTree(Data* data, int size, bool* usedFeatures, int featureCount) {
    // 新建结点
    TreeNode* node = malloc(sizeof(TreeNode));
    node->children = NULL;
    node->childFeatureValues = NULL;
    node->childrenCount = 0;
    node->featureValue = Nil;
    // 如果是纯净叶子结点，即已经为同一特征，无需再分，结束递归
    if (isPure(data, size)) {
        node->featureIndex = -1;
        node->classify = data[0].classify;
        return node;
    }
    // 寻找信息增益最大的特征及其信息增益值IG
    double bestGain = -1;
    Feature bestFeature = -1;
    for (int i = 0; i < featureCount; i++) {
        if (usedFeatures[i]) continue;
        double gain = calcGain(data, size, i, featureCount);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = i;
        }
    }
    // 未找到最佳特征，说明无法再分裂，直接返回叶子结点
    if (bestFeature == -1) {
        node->featureIndex = -1;
        node->classify = data[0].classify;
        return node;
    }
    // 找到最佳特征，创建结点
    node->featureIndex = bestFeature;
    node->classify = Nil;
    // 标记该特征已使用
    usedFeatures[bestFeature] = true;
    // 该特征的最大取值数
    int maxVal = getFeatureMaxValue(data, size, bestFeature);
    // 分配maxVal个孩子结点 定义决策树时有说过
    node->children = malloc(sizeof(TreeNode*) * maxVal);
    // 分析到这里了
    node->childFeatureValues = malloc(sizeof(int) * maxVal);
    for (int v = 1; v <= maxVal; v++) {
        Data* subset = malloc(sizeof(Data) * size);
        int subsetSize = splitData(data, size, bestFeature, v, subset,featureCount);
        if (subsetSize == 0) {
            node->children[v-1] = NULL;
            continue;
        }
        node->children[v-1] = buildTree(subset, subsetSize, usedFeatures, featureCount);
        node->children[v-1]->featureValue = v;
        node->childFeatureValues[v-1] = v;
        node->childrenCount++;
        for (int i = 0; i < subsetSize; i++) {
            free(subset[i].features);
        }
        free(subset);
    }
    // 清理
    usedFeatures[bestFeature] = false;
    return node;
}
// 打印决策树
void printTree(TreeNode* node, int depth) {
    for (int i = 0; i < depth; i++) printf("  ");
    if (node->featureIndex == -1) {
        printf("Leaf: %s\n", node->classify == Y ? "Yes" : "No");
    } else {
        printf("Feature %d\n", node->featureIndex);
        // int maxVal = getFeatureMaxValue(node->featureIndex);
        for (int i = 0; i < node->childrenCount; i++) {
            if (node->children[i] == NULL) continue;
            for (int j = 0; j < depth + 1; j++) printf("  ");
            printf("-- Value %d -->\n", node->childFeatureValues[i]);
            printTree(node->children[i], depth + 2);
        }
    }
}
// 分类
Type classifyByTree(TreeNode* root, Data* sample) {
    TreeNode* node = root;
    while (node->featureIndex != -1) {
        int val = sample->features[node->featureIndex];
        if (val < 1 || node->children[val - 1] == NULL) return Nil;
        node = node->children[val - 1];
    }
    return node->classify;
}
// 要进行改造，读取外存文件 完成对未知类别属性数据样例的分类
Data* readDataFile(const char* filename, bool isTrain, int* outSize, int* outFeatureCount) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    int capacity = 20;
    int size = 0;
    Data* data = malloc(sizeof(Data) * capacity);

    char line[MAX_LINE_LENGTH];
    int featureCount = -1;

    while (fgets(line, sizeof(line), file)) {
        int temp[MAX_FEATURES]; // 最多 64 个字段
        int count = 0;

        char* token = strtok(line, " \t\r\n");
        while (token && count < MAX_FEATURES) {
            temp[count++] = atoi(token);
            token = strtok(NULL, " \t\r\n");
        }

        // 跳过空行或格式错误
        if (isTrain) {
            if (count < 2) continue;
        } else {
            if (count < 1) continue;
        }

        int currentFeatureCount;
        if (isTrain) {
            currentFeatureCount = count - 1;
        } else {
            currentFeatureCount = count;
        }

        if (featureCount == -1) {
            featureCount = currentFeatureCount;
        } else if (featureCount != currentFeatureCount) {
            fprintf(stderr, "Feature Num is not equal %s On Line %d\n", filename, size + 1);
            exit(1);
        }
        // 扩容
        if (size >= capacity) {
            capacity *= 2;
            data = realloc(data, sizeof(Data) * capacity);
        }

        Data d;
        d.features = malloc(sizeof(int) * featureCount);
        for (int i = 0; i < featureCount; i++) {
            d.features[i] = temp[i];
        }
        if (isTrain) {
            d.classify = temp[count - 1];
        } else {
            d.classify = Nil;
        }
        data[size++] = d;
    }

    fclose(file);
    *outSize = size;
    *outFeatureCount = featureCount;
    return data;
}
// 导出为.dot
void exportDot(TreeNode* node, FILE* fp, int* nodeId, int parentId) {
    int currentId = (*nodeId)++;

    if (node->featureIndex == -1) {
        fprintf(fp, "  node%d [label=\"Leaf: %s\", shape=box];\n", currentId, node->classify == Y ? "Yes" : "No");
    } else {
        fprintf(fp, "  node%d [label=\"Feature %d\"];\n", currentId, node->featureIndex);
    }

    if (parentId != -1) {
        fprintf(fp, "  node%d -> node%d;\n", parentId, currentId);
    }

    if (node->featureIndex != -1) {
        for (int i = 0; i < node->childrenCount; ++i) {
            if (node->children[i]) {
                fprintf(fp, "  node%d -> node%d [label=\"=%d\"];\n", currentId, *nodeId, node->childFeatureValues[i]);
                exportDot(node->children[i], fp, nodeId, currentId);
            }
        }
    }
}

void writeDotFile(TreeNode* root, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to create dot file");
        return;
    }

    fprintf(fp, "digraph DecisionTree {\n");
    int nodeId = 0;
    exportDot(root, fp, &nodeId, -1);
    fprintf(fp, "}\n");

    fclose(fp);
}

int main() {
    int trainSize, trainFeatureCount;
    int testSize, testFeatureCount;
    // 待分类数据集
    Data* testData = readDataFile(TEST_FILE_PATH, false, &testSize, &testFeatureCount);
    // 训练数据集
    Data* trainData = readDataFile(TRAINING_FILE_PATH, true, &trainSize, &trainFeatureCount);
    
    if (trainFeatureCount != testFeatureCount) {
        fprintf(stderr, "Feature num is not equal\n");
        exit(1);
    }
    // 用于标记特征是否已使用 索引即为特征
    bool* usedFeatures = calloc(trainFeatureCount, sizeof(bool));
    // 建树
    TreeNode* root = buildTree(trainData, trainSize, usedFeatures,trainFeatureCount);
    // 打印树
    // printTree(root, 0);
    // 分类
    printf("\nTest Data Classification:\n");
    for (int i = 0; i < testSize; ++i) {
        Type result = classifyByTree(root, &testData[i]);
        printf("Test case %d => %s\n", i + 1, result == Y ? "Yes" : result == N ? "No" : "Unknown");
    }
    // 可视化图片
    writeDotFile(root, TREE_IMAGE_PATH);
    printf("Decision tree exported to %d\n",TREE_IMAGE_PATH);
    return 0;
}