#ifndef PREPROCESS_H_
#define PREPROCESS_H_

void reordershared(func *f1, func *f2);
void shared2most(func *f, chunk m);
void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2);

#endif  /* PREPROCESS_H_ */
