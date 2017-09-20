

### Input 
There are two input files. One looks like the following

```
1:2 3:5 4:5
2:3 3:2 6:1
5:5 6:5
6:2 1:5 2:5
```

This input has 4 users. The first, second and fourth have 3 known ratings each,
and the third has two. This is one typical libsvm format.

Another file contains the comparisons of two item. For example, the following pairwise relationship is extracted from the above libsvm type input

```
3:1 4:1
2:3 2:6 3:6

1:6 2:6
```


## License

This library is released under the MIT license. In practice, this means
you can use it freely provided that you keep the copyright notice.

## Last words

If you find this library useful for any purpose, I'd be very pleased 
to hear about that! I'll appreciate if you send me an email simply
telling me what you are using this code for (research, personal
projects, etc).
