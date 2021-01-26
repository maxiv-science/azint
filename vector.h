#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>

template <typename T>
class Vector
{
public:
    Vector();
    Vector(size_t size);
    ~Vector();
    void resize(size_t size);
    inline T& operator [] (size_t index) {return m_data[index];}
    inline const T& operator [] (size_t index) const 
    { 
        assert(index >= m_size);
        return m_data[index];
    }
    void push_back(const T& value);
    size_t size() const {return m_size;}
    size_t capacity() const {return m_capacity;}
    T* data() const {return m_data;}
    void leak() {m_leak = true;}
    void clear();
private:
    size_t m_size;
    size_t m_capacity;
    T* m_data;
    bool m_leak;
};



template<typename T> 
Vector<T>::Vector()
{
    m_data = nullptr;
    m_size = 0;
    m_capacity = 0;
    m_leak = false;
}

template<typename T> 
Vector<T>::Vector(size_t size)
{
    m_data = reinterpret_cast<T*>(malloc(size * sizeof(T)));
    m_size = size;
    m_capacity = size;
    m_leak = false;
}

template<typename T> 
Vector<T>::~Vector()
{
    if(!m_leak) {
        free(m_data);
    }
}
template<typename T> 
void Vector<T>::resize(size_t size)
{
    T* tmp = reinterpret_cast<T*>(realloc(m_data, size * sizeof(T)));
    if (tmp == nullptr) {
        printf("Error in realloc\n");
        exit(-1);
    }
    m_data = tmp;
    m_size = size;
    m_capacity = size;
}

template<typename T> 
void Vector<T>::push_back(const T& value)
{
    if ((m_size + 1) > m_capacity) {
        // growth factor is 1.5
        size_t new_capacity = (m_capacity*3)/2 + 8;
        T* tmp = reinterpret_cast<T*>(realloc(m_data, new_capacity * sizeof(T)));
        if (tmp == nullptr) {
            printf("Error in realloc\n");
            exit(-1);
        }
        m_data = tmp;
        m_capacity = new_capacity;
    }
    m_data[m_size] = value;
    m_size++;
}

template<typename T>
void Vector<T>::clear()
{
    free(m_data); 
    m_data = nullptr;
    m_size = 0; 
    m_capacity = 0;
}

