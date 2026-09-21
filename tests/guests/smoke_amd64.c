long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");
  __builtin_unreachable();}
void _start(void){
  unsigned char in[16];
  long n = sys_read(0, in, 16);
  if (n <= 0) sys_exit(1);
  unsigned long acc = 0;
  for (int i = 0; i < 16; i++) acc = (acc << 1) ^ in[i];
  sys_exit(acc & 1);
}
