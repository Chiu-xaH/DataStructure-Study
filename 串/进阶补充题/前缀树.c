#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#define ALPHABET_SIZE 26

typedef struct TrieNode {
    bool isEnd;   
    struct TrieNode* children[ALPHABET_SIZE];
} TrieNode, *Trie;

TrieNode* createNode() {
    TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
    node->isEnd = false;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        node->children[i] = NULL;
    }
    return node;
}

Trie Create() {
    return createNode();
}

// 插入单词
void Insert(Trie trie, char* word) {
    TrieNode* cur = trie;
    
    for (int i = 0; word[i] != '\0'; i++) {
        int idx = word[i] - 'a';
        if (cur->children[idx] == NULL) {
            cur->children[idx] = createNode();
        }
        // 下一个
        cur = cur->children[idx];
    }
    // 标记单词结束
    cur->isEnd = true;
}

// 查找完整单词
bool Search(Trie trie, char* word) {
    TrieNode* cur = trie;
    for (int i = 0; word[i] != '\0'; i++) {
        int idx = word[i] - 'a';
        if (cur->children[idx] == NULL) {
            // 未找到
            return false;
        }
        cur = cur->children[idx];
    }
    // 找到
    return cur->isEnd;
}

// 判断前缀
bool StartsWith(Trie trie, char* prefix) {
    TrieNode* cur = trie;
    for (int i = 0; prefix[i] != '\0'; i++) {
        int idx = prefix[i] - 'a';
        if (cur->children[idx] == NULL) {
            return false;
        }
        cur = cur->children[idx];
    }
    // 遍历结束后，说明前缀匹配成功
    return true;
}

void Free(Trie trie) {
    if (trie == NULL) return;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        Free(trie->children[i]);
    }
    free(trie);
}


int main() {
    Trie trie = Create();
    
    Insert(trie, "apple");
    Insert(trie, "app");
    
    printf("Search 'apple': %s\n", Search(trie, "apple") ? "true" : "false");
    printf("Search 'app': %s\n", Search(trie, "app") ? "true" : "false");
    printf("Search 'appl': %s\n", Search(trie, "appl") ? "true" : "false");
    
    printf("StartsWith 'app': %s\n", StartsWith(trie, "app") ? "true" : "false");
    printf("StartsWith 'ap': %s\n", StartsWith(trie, "ap") ? "true" : "false");
    
    Free(trie);
    
    return 0;
}