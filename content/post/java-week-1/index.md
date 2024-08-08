---
title: JAVA高级语言程序设计 - Week1概念整理
description: 课程第一周PPT知识点整理，完成于05.08
slug: java-w1
date: 2024-08-07 00:00:00+0000
# image: cover.jpg
categories:
    - 课业
tags:
    - Java
---

## Java 的历史

Sun 公司最早设计其运行在行动电话、PDA 上的编程语言，希望其是一门安全的语言，开发起来效率高。后面 PDA 失败了，重点转向了嵌入式软件系统，这时语言叫 Oak 语言。1993 年万维网出现，Mosaic 浏览器出现，该语言改名 Java，Mosaic 浏览器可以动态的下载 Java 代码运行（当时叫 applets），用作 Web 交互，允许用户在 Web 页面上打游戏、做表格之类的。

## Java 的特性

能做绝大多数其他传统主流语言可做的东西，但更简洁，更简单。

- 没有自动类型转换，是强类型语言
- 没有指针操作
- 没有 GOTO，没有全局变量，没有头文件
- 没有类似 C 语言的 struct 和 union（因为有面向对象了，没必要）
- 拥有自动垃圾回收机制（不需要手动 free 内存）
- 不允许多重继承
- 不兼容 C、C++

语言特性：

- 简单（我是没觉得，啰嗦倒是真啰嗦）
- 面向对象
- 平台不相关（一次编写到处运行）
- 鲁棒（拥有错误检查）
- 安全（拥有权限控制特性）
- 多线程
- 动态

## Java 程序的运行流程

Java 支持一次编写到处运行。Java 程序从编写到运行经历如下步骤：

1. 编写源代码，可以使用你喜欢的任意编辑器。
2. 通过指令 javac 将代码**编译**成**字节码**（xxx.class）（而非直接的可执行文件）
3. 通过指令 java 运行该字节码。会先检查这段字节码的所有字节是否合法，是否违反 java 安全限制，随后通过**解释器**在对应平台上的 Java 虚拟机（JVM，Java Virtual Machine）上执行该程序。

对于程序 MyProgram.java，应当先运行 `javac MyProgram.java`，随后运行 `java MyProgram`（注意没有.class），就可以运行程序了。

因为不同平台的指令集不同，Java 没有采取交叉编译，而是通过先编译成字节码再在对应平台通过平台自己的解释器执行，实现跨平台特性，即一次编写到处运行。
![](p/java-w1/assets/Pasted%20image%2020240507203433.png)

## Java 基础

这是一段最基础的 Java 代码：
```java
/**
* MyProgram.java
*
* Created on 28 June 2010, 17:56
*/
public class MyProgram {
	public static void main(String[] args) {
		System.out.println("Hello World!");
	}
}
```

### /** MyProgram.java...

Java 的注释和 C 比较类似：

- // 这是单行注释

- /\* 这是
- 多行
- 注释 \*/

长下面这样的叫 javadoc 注释。
Javadoc 这个工具能识别代码内的 Javadoc 注释，然后把他们整理成文档。
![](p/java-w1/assets/Pasted%20image%2020240507210728.png)

### public class MyProgram {

Java 对面向对象的崇拜已经到了痴狂的地步，所以每个程序的最根部应该是一个类，例如本程序的类名就是 MyProgram。（实际上在 Java21 还是 22 开始就允许你单写 main 开始，不过英方这边教的我建议还是按着 Java8 的规范来，虽然他确实要求用 OpenJDK21）
如果该类为 public，则要求**该代码的文件名必须等于类名**。例如此文件应当为 MyProgram.java。

### public static void main(String[] args) {

我们一行一行看：

- public：是指访问限制。public 说明该方法全局可访问。
- static：表示静态方法。即该方法为单例，与类相关，而非与对象相关。后面面向对象会讲。
- void：表示返回值类型。void 表示没有返回值。
- main：方法名。名为 main 的方法会被系统当作程序入口，和 C 类似。
- String[] args：指通过命令行传入的其他参数。详见 Lab 1.

### System.out.println("Hello World!");

这个东西类似 C 语言的 printf。ln 表示打印完了会带一个换行。

## Java 程序的编写、编译和运行

编写代码时，应当遵循 KISS 原则：Keep It Simple Stupid
JDK = Java Development Kit，开发套件，安装之后可以编译 Java 程序。
JRE = Java Runtime Environment，运行时环境，安装后可以运行 Java 程序。

现在基本上安装 JDK 时里面会带一份 JRE，不然光让开发不让运行也太蠢了。

一些基本的 JDK 指令：

- javac: compiler
- java: launcher for Java applications
- javadoc: API documentation generator
- jar: manages JAR files
- jdb: Java debugger

## 有关抄袭和剽窃

不要直接复制别人的代码。在你被允许使用他人代码时（例如 Mini Project 使用老师给的代码），应该写明 Javadoc 注释说明代码来源。

## Java 编程基础

最基本的程序结构模板如下：（暂时还没提到面向对象）

```java
class ClassName {
	public static void main(String[] args) {
		// 声明变量和方法
		// 写表达式...
	}
}
```

### 变量

Java 的变量可以在程序中的任何位置声明。变量声明方法与 C 语言类似。
```java
typeName name1, name2, ... namen;
typeName name1 = initvalue;
```
推荐的标识符命名方式：驼峰标记法。变量名使用小驼峰，类名使用大驼峰。

### 数据类型

Java 是强类型语言。意思是你声明变量时就需要指定该变量的数据类型。
基本数据类型：
![](p/java-w1/assets/Pasted%20image%2020240508131800.png)
![](p/java-w1/assets/Pasted%20image%2020240508131816.png)
注意：0.2363 类型的数据默认当成 double，0.2363F 才是单精度浮点；86827263927 默认当作 int，86827263927L 才会当作 long。

每种数据类型在 Java 中都有一个默认值，某些时候 Java 会把变量初始化为默认值。
字符串 String 不是 Java 的基本类型，而是一种对象。

#### 类型转换

Java 变量类型可自低到高自动无损转换。
```java
byte => short => int => long => float => double
```
反向转换时需要使用强制转换声明（Type Cast）：
```java
j = (int)(x + 1.3); // j = 8
i = (int)x + (int)1.3; // i = 7
```
反向转换时，由于变量范围问题，转换结果可能会被截断或溢出。

// TODO：写一份转换对照

### 保留字/关键字

这些词语不能当作用户标识符（变量名类名方法名等等）
![](p/java-w1/assets/Pasted%20image%2020240508132318.png)

### 赋值与操作符

基本用法：赋值符还是 `=`，自增自减运算符 `++` 和 `--` 也能用。

`++i` 和 `i++` 的效果和 C 语言也还是一样的，前者自身作为表达式会返回 `i+1`，后者自身作为表达式会返回 `i`，二者都会将 `i` 本身的值加一。
![](p/java-w1/assets/Pasted%20image%2020240508133057.png)

#### 代数操作符

加减乘除，取模（余数）。
![](p/java-w1/assets/Pasted%20image%2020240508133158.png)
所有代数操作符都可以与赋值符结合在一起用。
```java
int c = 3;
c += 7; // c = c + 7 = ?
c -= 5; // c = c - 5 = ?
c *= 6; // c = c * 6 = ?
c /= 3; // c = c / 3 = ?
c %= 3; // c = c % 3 = ?
```

#### 条件运算符（三元运算符）

比较特殊的一种运算符。
```java
a = (这个条件成立吗 ? 如果成立值就是我 : 否则就是我);
```

#### 操作符优先级

高优先级操作符优先运算，低优先级运算符后运算。
同优先级的运算符：二元操作符从左向右运算，赋值符从右向左运算。
![](p/java-w1/assets/Pasted%20image%2020240508134127.png)

#### 关系运算符/逻辑运算符

和 C 语言类似。
![](p/java-w1/assets/Pasted%20image%2020240508134415.png)
![](p/java-w1/assets/Pasted%20image%2020240508134439.png)

### 控制结构

选择结构：if、if else、switch
循环结构：while、do while、for
```java
if (这里的条件成立) {
	System.out.println("passed");
}
else {
	System.out.println("failed");
}
```
switch 结构和 C 类似，需要写 break，否则会一直执行下面的东西。
```java
char grade = 'a';
switch (grade) {
	case 'a':
		System.out.println("excellent");
		break;
	case 'b':
		System.out.println("good");
		break;
	case 'c':
		System.out.println("not bad");
		break;
	case 'd':
		System.out.println("bad");
		break;
	default:
		System.out.println("no such grade!");
}
```

for 循环跟 C 的写法很类似
```java
for (int i = 0; i < 3; i++) {
	System.out.println(“i = ” + i);
}
```

do while 会先做那段代码块里的东西，再判断条件如何。也就是不管什么情况都会先把代码块里的东西执行一次。while 用法和 C 语言类似。
``` java
do {
	System.out.println("i = " + i);
	i++;
} while (i < 3);

// 或者
int i = 0;
while (i < 3) {
	System.out.println("i = " + i);
	i++;
}
```

break：退出当前循环
continue：结束当前循环轮次
和 C 语言类似

Java 支持内层和外层循环标记，标记后可以在内层直接退出外层循环。
通过 outer inner 标记内层和外层循环，跳出（continue 或 break）时选择跳出外层循环。
如图：
![](p/java-w1/assets/Pasted%20image%2020240508163157.png)

## 面向对象基础

Java 是面向对象的编程语言，最基础的实体是**类（Class）**。

面向对象的设计方法将代码拆分为由类规划的对象。对象拥有自己的属性和方法。

例如，我要设计一个系统，该系统和动物、动物行为相关。
现在我要设计小猫。
小猫拥有小猫的**属性（Properties）**：例如毛色、年龄，名字、体重等等，这些数据是和这只小猫相关的，一只小猫可以拥有这些数据作为属性。
小猫拥有小猫的动作（**方法（Methods）**）：例如叫、吃东西、跑步等等。小猫可以拥有这些东西作为**函数方法**，也就是这只小猫的动作。

我们通过**编写一个类**来定义这个对象到底拥有什么。类就像是一个饼干模具，决定了饼干长什么样子，但是类本身并不是一个饼干。
一旦类编写好了，我们通过代码，**基于类来创建对象（Object）**。创建出的对象拥有这个类定义的全部内容。每个属性和方法现在都将**属于这个对象而非类**（除非声明为静态方法/属性）。

为了减少代码量，增加可维护性，类是可以**进行继承**的。例如，我的代码要拓展到动物，那么我定义一个类叫**动物**，让接下来的小猫、小狗都**继承这个动物**。这样，一旦后续需求变动（例如，需要为所有动物都添加“身高”的属性），那么我**直接修改动物类的属性**，继承了其的小猫小狗等**都会**拥有这个身高属性，就不需要一个一个修改这么多具体动物的类了。

面向对象更加灵活，还支持**多态、重载和重写**。

动物类实现了方法移动，但是狗可以跑，而兔子可以跳，这时狗和兔子就可以分别 **重写（Override）** 自己的移动方法，实现自己的逻辑。狗的跑只需要跑的距离作为参数，而兔子的跳需要跳的距离和高度两个参数，于是就可以在狗和兔子中分别**重载（Overload）** 这个方法，使其能够接收不同的参数，返回不同的内容，抛出不同的异常等等。
![](p/java-w1/assets/Pasted%20image%2020240508171141.png)
注意重载方法必须修改参数，不允许不动参数只动返回，那就报错了。
![](p/java-w1/assets/Pasted%20image%2020240508171647.png)

面向对象还支持**多态**。继承了动物的小猫既是动物，也是小猫。当我使用 Animal cat = new Cat() 将一个 Cat 送进 Animal 类型的变量时，他就既具有动物的特性也有小猫的特性，这里我们后面再聊。

还有更多特性：构造函数等等...
我们先有一个整体、宏观的印象，具体的东西我们后面再聊。
