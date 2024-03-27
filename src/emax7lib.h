
/* EMAX7 library                        */
/*         Copyright (C) 2013- by NAIST */
/*          Primary writer: Y.Nakashima */
/*                 nakashim@is.naist.jp */

/*******************************************************************************/
/******************************** Defs *****************************************/
/*******************************************************************************/
#ifndef __EMAX7LIB_H__
#define __EMAX7LIB_H__
#include "emax7.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <linux/ioctl.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { NANOS_ARM, NANOS_DRAIN, NANOS_CONF, NANOS_REGV, NANOS_RANGE, NANOS_LOAD, NANOS_EXEC, NANOS_TOTAL, NANOS_CLASS };

typedef struct {
  Uint  f : 23;
  Uint  e :  8;
  Uint  s :  1;
} f32bit;

typedef struct {
  Uint  e :  6;
  Uint  b :  1;
  Uint  s :  1; /* lower 8bit */
  Uint dm : 24; /* dummy for >gcc9 */
} wu7bit;

typedef struct {
  Uint  e :  7;
  Uint  s :  1; /* lower 8bit */
  Uint dm : 24; /* dummy for >gcc9 */
} wu8bit;

typedef struct {
  Uchar u[8];
} u64bit;

//#define abs(a)    ((a)>  0 ? (a) :-(a)    )
#define ad(a,b)   ((a)<(b)?(b)-(a):(a)-(b))
#define ss(a,b)   ((a)<(b)?   0   :(a)-(b))

/* dma_ctrl_space */
/* https://www.xilinx.com/content/dam/xilinx/support/documents/ip_documentation/axi_dma/v7_1/pg021_axi_dma.pdf */
struct dma_ctrl {
  /*   Register Name		   Address	Width	Type	Reset Value	Description */
  Uint MM2S_DMACR;             /* 0x00000000��  32      mixed   0x00010000      DMA Control register */
	/*   Field Name    Bits  Type Default Val  Description            */
	/*   Run/Stop         0  rw   0x0   ��	   0:stop-DMA,1:start-DMA */
	/*   Reserved 	      1  ro   0x1	   Reserved for future use */
        /*   Reset            2  rw   0x0          0:normal, 1:reset in progress */
        /*   Keyhole          3  rw   0x0          0:normal, 1:non-incrementing addr */
        /*   Cycle BD En      4  rw   0x0          0:normal, 1:Cycle Buffer Descriptor */
        /*   Reserved      11-5  ro   0x0          Reserved for future use */
        /*   IOC_IrqEn       12  rw   0x0          0:IOC   Intr. disabled, 1:IOC   Intr. enabled */
        /*   Dly_IrqEn       13  rw   0x0          0:Delay Intr. disabled, 1:Delay Intr. enabled */
        /*   Err_IrqEn       14  rw   0x0          0:Error Intr. disabled, 1:Error Intr. enabled */
        /*   Reserved        15  ro   0x0          Reserved for future use */
        /*   IRQThreshold 23-16  rw   0x1          Intr. threshold */
        /*   IRQThreshold 31-24  rw   0x0          Intr. delay time out */

  Uint MM2S_DMASR;             /* 0x00000004��  32      mixed   0x00010000      DMA Status register */
	/*   Field Name    Bits  Type Default Val  Description            */
	/*   Halted           0  ro   0x1   ��     0:DMA channel running, 1:DMA channel halted */
	/*   Idle             1  ro   0x0   ��     0:not Idle, 1:Idle */
        /*   Reserved         2  ro   0x0          Reserved for future use */
        /*   SGIncld          3  ro   C_INCLUDE_SG 0:SG N.A, 1:SG enabled */
        /*   DMAIntErr        4  ro   0x0   ��     0:no DMA internal error, 1:DMA internal error */
        /*   DMASlvErr        5  ro   0x0   ��     0:no DMA slave errors,   1:DMA slave error */
        /*   DMADecErr        6  ro   0x0   ��     0:no DMA decode errors,  1:DMA decode error (invalid address) */
        /*   Reserved         7  ro   0x0          Reserved for future use */
        /*   SGIntErr         8  ro   0x0          0:no SG internal error,  1:SG internal error */
        /*   SGSlvErr         9  ro   0x0          0:no SG slave errors,    1:SG slave error */
        /*   SGDecErr        10  ro   0x0          0:no SG decode errors,   1:SG decode error (invalid address) */
        /*   Reserved        11  ro   0x0          Reserved for future use */
        /*   IOC_Irq         12  rwc  0x0          0:no IOC intr.   1:IOC intr. */
        /*   Dly_Irq         13  rwc  0x0          0:no Delay intr. 1:Delay intr. */
        /*   Err_Irq         14  rwc  0x0          0:no Err intr.   1:Err intr. */
        /*   Reserved        15  ro   0x0          Reserved for future use */
        /*   IRQThreshold 23-16  ro   0x1          Intr. threshold stat */
        /*   IRQThreshold 31-24  ro   0x0          Intr. delay time out stat */

  Uint reserved0[4];           /* 08h - 14h Reserved N/A */
  Uint MM2S_SA;                /* 0x00000018    32      rw      0x00000000      Source Address. Lower 32 bits of address.*/
  Uint MM2S_SA_MSB;            /* 0x0000001c    32      rw      0x00000000      Source Address. Upper 32 bits of address.*/
  Uint reserved1[2];           /* 20h - 24h Reserved N/A */
  Uint MM2S_LENGTH;            /* 0x00000028    32      rw      0x00000000      Transfer Length (Bytes) */
  Uint reserved2[1];           /* 2ch       Reserved N/A */
  Uint S2MM_DMACR;             /* 0x00000030��  32      mixed   0x00010000      DMA Control register */
  Uint S2MM_DMASR;             /* 0x00000034��  32      mixed   0x00010000      DMA Status register */
  Uint reserves3[4];           /* 38h - 44h Reserved N/A */
  Uint S2MM_DA;                /* 0x00000048    32      rw      0x00000000      Destination Address. Lower 32 bit address.*/
  Uint S2MM_DA_MSB;            /* 0x0000004c    32      rw      0x00000000      Destination Address. Upper 32 bit address.*/
  Uint reserved4[2];           /* 50h - 54h Reserved N/A */
  Uint S2MM_LENGTH;            /* 0x00000058    32      rw      0x00000000      Buffer Length (Bytes) */

  /* Simple Mode */
  /* 0. MM2S_DMASR.Halted=0,Idle=1���ǧ */
  /* 1. Start the MM2S channel (MM2S_DMACR.RS = 1).*/
  /* 2. Write a valid source address to MM2S_SA+MM2S_SA_MSB register.*/
  /* 3. Write the bytes to MM2S_LENGTH register. A value of zero has no effect.*/

  /* 0. S2MM_DMASR.Halted=0,Idle=1���ǧ */
  /* 1. Start the S2MM channel (S2MM_DMACR.RS = 1).*/
  /* 2. Write a valid destination address to S2MM_DA+S2MM_DA_MSB register.*/
  /* 3. Write the bytes to S2MM_LENGTH register. A value of zero has no effect.*/

  /* 4. MM2S_DMASR.bit4-6!=0�ʤ饨�顼 */
  /* 4. S2MM_DMASR.bit4-6!=0�ʤ饨�顼 */
  /* 4. MM2S_DMASR.IOC_Irq��1�ˤʤ�ޤ��Ե�,1��񤤤�reset */
  /* 4. S2MM_DMASR.IOC_Irq��1�ˤʤ�ޤ��Ե�,1��񤤤�reset */
};

/* reg_ctrl */
enum { EXRING_IDLE, EXRING_BUSY};
enum { LMRING_IDLE, LMRING_BUSY};
enum { CMD_NOP, CMD_RESET, CMD_SCON, CMD_EXEC};
struct reg_ctrl {
  struct i0 {
    Ull  stat; /* +0000 bit15-12:LMM_SIZE, bit11-8:EMAX_DEPTH, bit7-4:LMRING, bit3-0:EXRING */
    Uint mcid; /* +0008 maximum chip-ID of IMAX (<EMAX_NCHIP) to be chained (activated) */
    Uint dmy0;
    Uint cmd;  /* +0010 host writes Ull cmd then chip# is propagated to succesors */
  /*Uint cid;*//* +0012 chip# ( set by write to cmd ) */
    Uint dmy1;
    Ull  dmy2;
    Ull  adtr; /* +0020 */
    Ull  dmy3;
    Ull  csel; /* +0030 */
    Ull  dmrp; /* +0038 DMAREAD-PREF */
    Ull  dmy4[1016];
    struct conf                    conf[AMAP_DEPTH][EMAX_WIDTH];  /* +2000-3fff */
    struct {Ull  br[UNIT_WIDTH];}  breg[AMAP_DEPTH][EMAX_WIDTH];  /* +4000-5fff *//* unit[cid][EMAX_DEPTH].breg[x][EMAX_WIDTH].br[UNIT_WIDTH] is used */
    struct {Uint ea0b ; /* ea0 base   (for avoiding ld-mask-st, should be extended to 32bits (lower 18bit is available)) */
      /*Ull  dmy0 :14;*/
        Uint ea0o ; /* ea0 offset (for avoiding ld-mask-st, should be extended to 32bits (lower 18bit is available)) */
      /*Ull  dmy1 :14;*/
        Uint ea1b ; /* ea1 base   (for avoiding ld-mask-st, should be extended to 32bits (lower 18bit is available)) */
      /*Ull  dmy2 :14;*/
        Uint ea1o ; /* ea1 offset (for avoiding ld-mask-st, should be extended to 32bits (lower 18bit is available)) */
      /*Ull  dmy3 :14;*/
        Uint top  ; /* LMM-top virtual-address */
      /*Ull  dmy4 : 1;*/
        Uint bot  ; /* LMM-bot virtual-address */
      /*Ull  dmy5 : 1;*/
        Ull  dmy6 ;}           addr[AMAP_DEPTH][EMAX_WIDTH];  /* +6000-7fff */
    struct {Ull  reg[UNIT_WIDTH];} lddmrw[AMAP_DEPTH][EMAX_WIDTH];/* +8000-9fff *//* lddmq/trans-r,lddmq-w */
    Ull dmy5[3072]; /* +a000-ffff */
  } i[EMAX_NCHIP]; /* 0000-ffff */
};

/* emax7 host control */
enum { STATUS_IDLE, STATUS_CONF, STATUS_SCON, STATUS_REGV, STATUS_RANGE, STATUS_DRAIN, STATUS_LOAD, STATUS_START, STATUS_EXEC, STATUS_TERM };

pthread_mutex_t axi_dma_mutex;

struct emax7 { /* host status of EMAX7 */
  volatile Ull   dma_ctrl;  /* struct dma_ctrl *dma_ctrl  DMA control */
  volatile Ull   reg_ctrl;  /* struct reg_ctrl *reg_ctrl  REG control */

  Ull   status            : 4;
  Ull   csel_save         : 2;
  Ull   last_conf            ; /* for insn_reuse */
  Ull   lmmic             : 1; /* 0:lmm[0] is curent, 1:lmm[1] is current */
  Ull   lmmio             : 1; /* 0:lmm[0] is prev,   1:lmm[1] is prev    */
  Ull   mapdist           : 6; /* specified mapdist */
  Ull   lastdist          : 6; /* lastdist for DYNAMIC_SCON */
  struct lmmi lmmi[EMAX_NCHIP][AMAP_DEPTH][EMAX_WIDTH][2]; /* lmmi for host (len/ofs/top are resolved) */
  Ull   lmmi_bitmap[EMAX_WIDTH];      /* based on lmmi[*][EMAX_WIDTH][2].v */
  Uchar lmmd[AMAP_DEPTH][EMAX_WIDTH]; /* chip#7,6,..,0:clean, 1:dirty, exec��store�ս��1, drainľ��0 */

#ifndef IGNORE_LMMI_BLKGATHER
  Ull   plist                ; /* pointer-list */
  Ull   blkcount          : 7; /* active block number */
  Ull   blksize           : 9; /* 1:64 2:128 3:256 dwords */
  Ull   lmmblktop            ; /* LMM-addr     for LDRQ(blk>0) */
  Ull   lmmblklen            ; /* total dwords for LDRQ(blk>0) */
#endif

  Ull   rw                    ; /* 0:load(mem->lmm), 1:store(lmm->mem)      */
  Ull   ddraddr               ; /* ddr-address                              */
  Ull   lmmaddr               ; /* lmm-address                              */
  Ull   dmalen                ; /* dma-length                               */
  Ull   sigwait               ; /* 0:no macropipe+sigwait, 1:macropipe+sigwait */
  int   *sigstat              ; /* ->th_args.stat (0:idle, 1:run, 2:wait)   */
  sigset_t *sigset            ; /* for sigmask/sigwait                      */

#ifndef IGNORE_LDDMQ_HANDSHAKE
  Ull   fsm_busy          : 1; /* for LDDMQ and TR */
  Ull   lmwd_valid        : 1; /* for LDDMQ */
  Ull   tcureg_valid      : 1; /* fsm->ARM   0 -> 1 -> 1 -> 0 -> 0 -> 0                              */
  Ull   tcureg_ready      : 1; /* fsm<-ARM   0 -> 0 -> 1 -> 0 -> 0 -> 0                              */
  Ull   tcureg_last       : 1; /* fsm->ARM   0 -> 0 -> 0 -> 1 -> 1 -> 0                              */
  Ull   tcureg_term       : 1; /* fsm<-ARM   0 -> 0 -> 0 -> 0 -> 1 -> 0                              */
  Ull   tcureg[UNIT_WIDTH]   ; /* tcu-data        of tcu                       v                     */
                               /* from ARM:  svc 0x1010 ... tcureg_valid->x0                         */
                               /* from ARM:  svc 0x1011 ... 1->tcureg_ready                          */
                               /* from ARM:  svc 0x1012 ... tcureg_last->x0                          */
                               /* from ARM:  svc 0x1013 ... 1->tcureg_term                           */
                               /* from ARM:  svc 0x1014 ... tcureg[3:0]->x3,2,1,0                    */
#endif
} emax7[EMAX_NLANE];

volatile struct emax_info {
  Ull  dma_phys;     // kern-phys
  Ull  dma_vadr;     // not used
  Ull  dma_mmap;     // user-virt Contiguous 64K register space
  Ull  reg_phys;     // kern-phys
  Ull  reg_vadr;     // not used
  Ull  reg_mmap;     // user-virt Contiguous 4GB space including LMM space
  Ull  lmm_phys;     // kern-phys
  Ull  lmm_vadr;     // not used
  Ull  lmm_mmap;     // user-virt Contiguous 2GB space for LMM space
  Ull  ddr_phys;     // kern-phys
  Ull  ddr_vadr;     // not used
  Ull  ddr_mmap;     // user-virt Contiguous 2GB space in DDR-high-2GB space
  int  driver_use_1;
  int  driver_use_2;
} emax_info[EMAX_NLANE];

/*  ... for ARMSIML only */
#define DMA_BASE2_PHYS	 0x50000000
#define REG_BASE2_PHYS	 0x50100000
#define REG_CONF_OFFS    0x00002000
#define REG_BREG_OFFS    0x00004000
#define REG_ADDR_OFFS    0x00006000
#define REG_LDDM_OFFS    0x00008000
#define REG_AREA_MASK    0x0000ffff
#define LMM_BASE2_PHYS 	 0x60000000
#define MEM_VALID_ADDR	 0xafffffff

#ifdef ARMZYNQ
/*******************************************************************************/
/******************************** ZYNQ-COMMON **********************************/
/*******************************************************************************/

#define DMA_BASE_PHYS	 0x00000000a4000000LL
#define DMA_BASE_PHYSOFS 0x0000000000010000LL
#define DMA_MMAP_SIZE	 0x0000000000010000LL
/*  ... 64KB  */
#define REG_BASE_PHYS	 0x0000020800000000LL
#define REG_BASE_PHYSOFS 0x0000000800000000LL
#define REG_MMAP_SIZE	 0x0000000200000000LL
/*  ... 8GB REGS(2G)+LMM(4G) */
#define LMM_BASE_PHYS	 0x0000020880000000LL
#define LMM_BASE_PHYSOFS 0x0000000800000000LL
/*  ... 4GB   */
#define DDR_BASE_PHYS	 0x0000050000000000LL
#define DDR_MMAP_SIZE	 0x0000000100000000LL
/*  ... 4GB   */

#define EMAX_IOC_MAGIC  60
/* Please use a different 8-bit number in your code */
#define EMAX_IORESET			_IO( EMAX_IOC_MAGIC, 0)
#define EMAX_GET_ACPMEM			_IOWR(EMAX_IOC_MAGIC,  1, unsigned long)
#define EMAX_IOC_MAXNR 2

static int filter(struct dirent *dir)
{
  return dir->d_name[0] == '.' ? 0 : 1;
}

static void trim(char *d_name)
{
  char *p = strchr(d_name, '\n');
  if (p != NULL) *p = '\0';
}

static int is_target_dev(char *d_name, char *target)
{
  char path[32];
  char name[32];
  FILE *fp;
  sprintf(path, "/sys/class/uio/%s/name", d_name);
  if ((fp = fopen(path, "r")) == NULL) return 0;
  if (fgets(name, sizeof(name), fp) == NULL) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  if (strcmp(name, target) != 0) return 0;
  return 1;
}

static int get_reg_size(char *d_name)
{
  char path[32];
  char size[32];
  FILE *fp;
  sprintf(path, "/sys/class/uio/%s/maps/map0/size", d_name);
  if ((fp = fopen(path, "r")) == NULL) return 0;
  if (fgets(size, sizeof(size), fp) == NULL) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  return strtoull(size, NULL, 16);
}

emax7_open(int NLANE)
/* HPM���ͳ��������쥸�����˥ꥻ�å����� */
/* HPP���ͳ�������������۶��֤˼��� */
/* ACP���ͳ����conf/lmmi/regv���֤��۶��֤˼��� */
{
  struct dirent **namelist;
  int  num_dirs, dir, uiolen;
  int  reg_size;
  char path[1024];
  int  fd_dma;
  int  fd_reg;
  int  fd_ddr;
  char *UIO_AXI_C2C       = "axi_chip2chip\n";
  char *UIO_AXI_MM2S      = "axi_mm2s_mapper\n";
  char *UIO_DMA           = "dma\n";
  char *UIO_AXI_EMAX6     = "emax6\n";
  char *UIO_DDR_HIGH      = "ddr_high\n";
  int  fd_dma_found = 0;
  int  emax7_found = 0;
  int  i;

  pthread_mutex_init(&axi_dma_mutex, NULL);

  if ((num_dirs = scandir("/sys/class/uio", &namelist, filter, alphasort)) == -1)
    return (NULL);

  for (dir = 0; dir < num_dirs; ++dir)
    trim(namelist[dir]->d_name);
 
  for (uiolen=4; uiolen<6; uiolen++) { /* /dev/uioX -> /dev/uio1X ... */
    for (dir = 0; dir < num_dirs; ++dir) {
      if (strlen(namelist[dir]->d_name)!=uiolen) /* ignore /dev/uio1X */
	continue;
      if (is_target_dev(namelist[dir]->d_name, UIO_AXI_EMAX6)) {
	sprintf(path, "/dev/%s", namelist[dir]->d_name);
	if ((fd_reg = open(path, O_RDWR | O_SYNC)) == -1) {
	  printf("open failed. %s", UIO_AXI_EMAX6);
	  return (NULL);
	}
	printf("%s: %s", path, UIO_AXI_EMAX6);
	if (emax7_found >= EMAX_NLANE || emax7_found >= NLANE) {
	  printf("emax7_found > EMAX_NLANE || emax7_found >= given_NLANE (skip)\n");
	  continue; /* skip rest of EMAX7 */
	}
	// mmap(cache-off) 4KB aligned
	emax_info[emax7_found].reg_phys = REG_BASE_PHYS+REG_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].reg_mmap = (Ull)mmap(NULL, REG_MMAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd_reg, 0); /* 4GB */
	close(fd_reg);
	if (emax_info[emax7_found].reg_mmap == MAP_FAILED) {
	  printf("fd_reg mmap() failed. errno=%d\n", errno);
	  return (NULL);
	}
	emax_info[emax7_found].lmm_phys = LMM_BASE_PHYS+LMM_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].lmm_mmap = emax_info[emax7_found].reg_mmap + (LMM_BASE_PHYS - REG_BASE_PHYS);
	emax7_found++;
      }
    }
  }
  for (uiolen=4; uiolen<6; uiolen++) {
     for (dir = 0; dir < num_dirs; ++dir) {
      if (strlen(namelist[dir]->d_name)!=uiolen) /* ignore /dev/uio1X */
	continue;
      if (is_target_dev(namelist[dir]->d_name, UIO_DMA) && (reg_size = get_reg_size(namelist[dir]->d_name))) {
	sprintf(path, "/dev/%s", namelist[dir]->d_name);
	if ((fd_dma = open(path, O_RDWR | O_SYNC)) == -1)
	  continue;
	printf("%s: %s", path, UIO_DMA);
	if (fd_dma_found >= EMAX_NLANE || fd_dma_found >= NLANE) {
	  printf("fd_dma_found > EMAX_NLANE || fd_dma_found > given_NLANE (skip)\n");
	  continue; /* skip rest of FDDMA */
	}
	emax_info[fd_dma_found].dma_phys = DMA_BASE_PHYS+DMA_BASE_PHYSOFS*fd_dma_found;
	emax_info[fd_dma_found].dma_mmap = (Ull)mmap(NULL, reg_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd_dma, 0);
	close(fd_dma);
	if (emax_info[fd_dma_found].dma_mmap == MAP_FAILED)
	  continue;
	fd_dma_found++;
      }
      else if (is_target_dev(namelist[dir]->d_name, UIO_AXI_C2C)) {
	sprintf(path, "/dev/%s", namelist[dir]->d_name);
	if ((fd_reg = open(path, O_RDWR | O_SYNC)) == -1) {
	  printf("open failed. %s", UIO_AXI_C2C);
	  return (NULL);
	}
	printf("%s: %s", path, UIO_AXI_C2C);
	if (emax7_found >= EMAX_NLANE || emax7_found >= NLANE) {
	  printf("emax7_found > EMAX_NLANE || emax7_found > given_NLANE (skip)\n");
	  continue; /* skip rest of EMAX7 */
	}
	// mmap(cache-off) 4KB aligned
	emax_info[emax7_found].reg_phys = REG_BASE_PHYS+REG_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].reg_mmap = (Ull)mmap(NULL, REG_MMAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd_reg, 0); /* 4GB */
	close(fd_reg);
	if (emax_info[emax7_found].reg_mmap == MAP_FAILED) {
	  printf("fd_reg mmap() failed. errno=%d\n", errno);
	  return (NULL);
	}
	emax_info[emax7_found].lmm_phys = LMM_BASE_PHYS+LMM_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].lmm_mmap = emax_info[emax7_found].reg_mmap + (LMM_BASE_PHYS - REG_BASE_PHYS);
	emax7_found++;
      }
      else if (is_target_dev(namelist[dir]->d_name, UIO_AXI_MM2S)) {
	sprintf(path, "/dev/%s", namelist[dir]->d_name);
	if ((fd_reg = open(path, O_RDWR | O_SYNC)) == -1) {
	  printf("open failed. %s", UIO_AXI_MM2S);
	  return (NULL);
	}
	printf("%s: %s", path, UIO_AXI_MM2S);
	if (emax7_found >= EMAX_NLANE || emax7_found >= NLANE) {
	  printf("emax7_found > EMAX_NLANE || emax7_found > given_NLANE (skip)\n");
	  continue; /* skip rest of EMAX7 */
	}
	// mmap(cache-off) 4KB aligned
	emax_info[emax7_found].reg_phys = REG_BASE_PHYS+REG_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].reg_mmap = (Ull)mmap(NULL, REG_MMAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd_reg, 0); /* 4GB */
	close(fd_reg);
	if (emax_info[emax7_found].reg_mmap == MAP_FAILED) {
	  printf("fd_reg mmap() failed. errno=%d\n", errno);
	  return (NULL);
	}
	emax_info[emax7_found].lmm_phys = LMM_BASE_PHYS+LMM_BASE_PHYSOFS*emax7_found;
	emax_info[emax7_found].lmm_mmap = emax_info[emax7_found].reg_mmap + (LMM_BASE_PHYS - REG_BASE_PHYS);
	emax7_found++;
      }
      else if (is_target_dev(namelist[dir]->d_name, UIO_DDR_HIGH)) {
	sprintf(path, "/dev/%s", namelist[dir]->d_name);
	if ((fd_ddr = open(path, O_RDWR | O_SYNC)) == -1) {
	  printf("open failed. %s",UIO_DDR_HIGH);
	  return (NULL);
	}
	printf("%s: %s", path, UIO_DDR_HIGH);
	// mmap(cache-on)  4KB aligned
	emax_info[0].ddr_phys = DDR_BASE_PHYS;
	emax_info[0].ddr_mmap = (Ull)mmap(NULL, DDR_MMAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd_ddr, 0); /* 2GB */
	close(fd_ddr);
	if (emax_info[0].ddr_mmap == MAP_FAILED) {
	  printf("fd_ddr mmap() failed. errno=%d\n", errno);
	  return (NULL);
	}
      }
    }
  }

  for (dir = 0; dir < num_dirs; ++dir)
    free(namelist[dir]);
  free(namelist);

  if (!emax7_found) {
    printf("EMAX not found: %s", UIO_AXI_EMAX6);
    exit(1);
  }
  if (fd_dma_found != emax7_found) {
    printf("Warning: fd_dma_found(%d) != emax7_found(%d)\n", fd_dma_found, emax7_found);
    if (fd_dma_found < emax7_found)
      emax7_found = fd_dma_found;
    else
      fd_dma_found = emax7_found;
  }

  for (i=0; i<fd_dma_found; i++) {
    ((struct dma_ctrl*)emax_info[i].dma_mmap)->MM2S_DMACR = 0x00010004;
    ((struct dma_ctrl*)emax_info[i].dma_mmap)->MM2S_DMASR = 0x00017000;
    ((struct dma_ctrl*)emax_info[i].dma_mmap)->S2MM_DMACR = 0x00010004;
    ((struct dma_ctrl*)emax_info[i].dma_mmap)->S2MM_DMASR = 0x00017000;
  }
  for (i=1; i<emax7_found; i++) {
    emax_info[i].ddr_phys = emax_info[0].ddr_phys;
    emax_info[i].ddr_mmap = emax_info[0].ddr_mmap;
  }

  return (emax7_found);
}
#endif

/*******************************************************************************/
/******************************** Timer ****************************************/
/*******************************************************************************/

Ull nanosec_sav[EMAX_NLANE];
Ull nanosec[EMAX_NLANE][NANOS_CLASS];

sleep_nanosec(int nano)
{
#if defined(ARMSIML)
#else
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = nano;
  nanosleep(NULL, &ts);
#endif
}

reset_nanosec(int LANE)
{
  int i;
  for (i=0; i<NANOS_CLASS; i++)
    nanosec[LANE][i] = 0;
#if defined(ARMSIML)
  nanosec_sav[LANE] = _getclk(0);
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_sav[LANE] = 1000000000*ts.tv_sec + ts.tv_nsec;
#endif
}

get_nanosec(int LANE, int class)
{
  Ull nanosec_now;
#if defined(ARMSIML)
  nanosec_now = _getclk(0);
  nanosec[LANE][class] += nanosec_now - nanosec_sav[LANE];
  nanosec[LANE][NANOS_TOTAL] += nanosec_now - nanosec_sav[LANE];
  nanosec_sav[LANE] = nanosec_now;
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_now = 1000000000*ts.tv_sec + ts.tv_nsec;
  nanosec[LANE][class] += nanosec_now - nanosec_sav[LANE];
  nanosec[LANE][NANOS_TOTAL] += nanosec_now - nanosec_sav[LANE];
  nanosec_sav[LANE] = nanosec_now;
#endif
}

show_nanosec(int LANE)
{
#if defined(ARMSIML)
  printf("LANE%d SIML_cycle/1000: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
	 LANE,
	 (Uint)(nanosec[LANE][NANOS_ARM]/1000),
	 (Uint)(nanosec[LANE][NANOS_DRAIN]/1000),
	 (Uint)(nanosec[LANE][NANOS_CONF]/1000),
	 (Uint)(nanosec[LANE][NANOS_REGV]/1000),
	 (Uint)(nanosec[LANE][NANOS_RANGE]/1000),
	 (Uint)(nanosec[LANE][NANOS_LOAD]/1000),
	 (Uint)(nanosec[LANE][NANOS_EXEC]/1000),
	 (Uint)(nanosec[LANE][NANOS_TOTAL]/1000));
#else
  printf("LANE%d usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
	 LANE,
	 (Uint)(nanosec[LANE][NANOS_ARM]/1000),
	 (Uint)(nanosec[LANE][NANOS_DRAIN]/1000),
	 (Uint)(nanosec[LANE][NANOS_CONF]/1000),
	 (Uint)(nanosec[LANE][NANOS_REGV]/1000),
	 (Uint)(nanosec[LANE][NANOS_RANGE]/1000),
	 (Uint)(nanosec[LANE][NANOS_LOAD]/1000),
	 (Uint)(nanosec[LANE][NANOS_EXEC]/1000),
	 (Uint)(nanosec[LANE][NANOS_TOTAL]/1000));
#endif
}

/*******************************************************************************/
/******************************** DMA-START ************************************/
/*******************************************************************************/

#if !defined(EMAXNC)
emax7_check_lmmi_and_dma(int LANE, int mode, int phase, int lastdist, int c, int i, int j)
{
  /* mode   0:array, 1:drain */
  /* phase  0:nop,   1:drain, 2:load, 3exec */
  /* lastdist */
  /* i      row              */
  /* j      col              */
  /* lmmi������˸ƤФ��. lmd��Ϣ��lmd���֤����lmr/lmx�򸡺�(+lastdist����ͳ) */
  /*                   ����lmd    ���֤�,"lmmi[i         ][lmmic]" */
  /* lastdist=>0�ξ��,����lmw/lmx���֤�,"lmmi[i+lastdist][lmmio]" */
  /* ����,lmd�ξ��,SCON���Ƥ⤷�ʤ��Ƥ�EXEC��Ʊ����DRAIN (lastdist=0�ξ���lmm��Ⱦʬ��lmd/lmw�˻Ȥ�ʬ��������) */
  /* lastdist=0�ξ��,DYNAMIC_SCON��̵��̣�ʤΤǽ����̤� */
  /* lastdist>0�ξ��,DYNAMIC_SCON����SCONͭ��Ͻ����̤� */
  /*                 SCON̵��ξ��,���⤽��lmd���֤����ʤ��Τ�lmd�����̵�뤹�٤�.��lmw/lmx��EXEC��DRAIN��ɬ�� */
  /*                 ������������,lmd��Ȥ��������Ǥ�DYNAMIC_SCON��Ȥ�ʤ��Ϥ��ʤΤ�,�����б����ʤ� */
  int k, m = (i+lastdist)%EMAX_DEPTH; /* lmmo-index */
  int lmmc_topz;
  int lmmc_ofsz;
  int lmmo_stat;
  int lmmc_stat;
  int lmm_ready;
  int lmm_readz;
  int mark;

  struct lmmi *lmmiop  = &emax7[LANE].lmmi[c][m][j][emax7[LANE].lmmio];
  struct lmmi *lmmicp  = &emax7[LANE].lmmi[c][i][j][emax7[LANE].lmmic];
  struct lmmi *lmmiop1 = &emax7[LANE].lmmi[c][(m+1)%EMAX_DEPTH][j][emax7[LANE].lmmio];
  struct lmmi *lmmicp1 = &emax7[LANE].lmmi[c][(i+1)%EMAX_DEPTH][j][emax7[LANE].lmmic];

  Ull dmadr;
  int dmlen;
  Ull dmnxt;
  int dmrw; /* 0:mem->lmm 1:lmm->mem */
  static Ull concat_adr[EMAX_NLANE][EMAX_NCHIP]; /* NULL:invalid, !NULL:top_addr */
  static int concat_len[EMAX_NLANE][EMAX_NCHIP]; /* byte-len */

  /* check_lmmi */
  if ((phase == 1 && mode == 0) || phase == 2 || phase == 3) { /* (drain && array) || load || exec */
    lmmc_topz = (lmmicp->top == 0);
    lmmc_ofsz = (lmmicp->ofs == 0);
    lmmo_stat = (lmmiop->v<<3)|(lmmiop->rw<<2)|(lmmiop->f<<1)|(lmmiop->p); /* v|rw|f|p */
    lmmc_stat =((lmmicp->v & ~lmmicp->hcopy & ~lmmicp->vcopy & ((lmmicp->f&lmmicp->p) | !lmmc_topz))<<3)|(lmmicp->rw<<2)|(lmmicp->f<<1)|(lmmicp->p); /* v= ~copy & (OP_LDDMQ/OP_TR �ޤ��� ptop!=NULL) */
    lmm_ready = (lmmiop->v && lmmiop->blk == lmmicp->blk && lmmiop->len == lmmicp->len && lmmiop->top == lmmicp->top);
    lmm_readz = (lmmiop->v && lmmiop->blk == lmmicp->blk && lmmiop->len == lmmicp->len &&(lmmiop->top+(Sll)(int)lmmiop->ofs) == lmmicp->top);
  }

  /* lmx: bitmap�򸡺���,��addr+len�ȼ�addr�����,Ϣ³�ʤ�Ϣ�뤷����addr/len����¸.�ǽ��ޤ�����Ϣ³�ʤ���¸addr/len�ޤ��ϸ�addr/len��Ȥä�DMA */

  if      (phase == 1) { /* drain */
    if      (mode==0 && lmmo_stat==12 && lmmc_stat!=13 && (emax7[LANE].lmmd[m][j]&1<<c)) { mark=1;emax7[LANE].lmmd[m][j]&=~(1<<c);dmadr=lmmiop->top;dmlen=lmmiop->len;dmnxt=lmmiop1->top;dmrw=1;}/* ��2 lmw&!lmd drain */
    else if (mode==0 && lmmo_stat==14 && !lmm_ready    && (emax7[LANE].lmmd[m][j]&1<<c)) { mark=1;emax7[LANE].lmmd[m][j]&=~(1<<c);dmadr=lmmiop->top;dmlen=lmmiop->len;dmnxt=lmmiop1->top;dmrw=1;}/* ��4 lmx      drain */
    else if (mode==1 &&                                   (emax7[LANE].lmmd[i][j]&1<<c)) { mark=1;emax7[LANE].lmmd[i][j]&=~(1<<c);dmadr=lmmicp->top;dmlen=lmmicp->len;dmnxt=lmmicp1->top;dmrw=1;}/* �� drain_dirty_lmm */
    else                                                                              { mark=0;                                                                                        }
  }
  else if (phase == 2) { /* load */
    if     ((lmmc_stat== 8               && !lmm_ready)                                                                                                                              /* ��1 lmr & !ready */
         || (lmmc_stat== 9               && !lmm_readz)                                                                                                                              /* ��7 lmp & !readz */
         || (lmmc_stat==10                            )                                                                                                                              /* ��3 lmf always load */
         || (lmmc_stat==14               && !lmm_ready))                           { mark=1;                          dmadr=lmmicp->top;dmlen=lmmicp->len;dmnxt=lmmicp1->top;dmrw=0;}/* ��3 lmx always load */
    else                                                                           { mark=0;                                                                                        }/* skip load */
  }
  else if (phase == 3) { /* exec */
    if      (lmmc_stat== 9 && (lastdist||!lmmc_ofsz)) { mark=1;                                                             dmadr=lmmicp->top;dmlen=lmmicp->len;dmrw=0;             }/* ��5 lmp */
    else if (lmmc_stat==12 || lmmc_stat==14         ) { mark=0;emax7[LANE].lmmd[i][j]|=(1<<c);                                                                                      }/* ��6 lmw/lmx */
    else if (lmmc_stat==13                          ) { mark=  emax7[LANE].lmmd[m][j]& (1<<c); emax7[LANE].lmmd[m][j]|=((!lastdist)<<c);dmadr=lmmicp->top;dmlen=lmmicp->len;dmrw=1; }/* ��6 lmd & dirty */
#ifndef IGNORE_LDDMQ_HANDSHAKE
//  else if (lmmc_stat==11                          ) { mark=1;                             } /*     LDDMQ */
//  else if (lmmc_stat==15                          ) { mark=1;                             } /*     TR */
#endif
    else                                              { mark=0;                             } /* skip pdrain/pload */
  }

  if (mark) {
#if 1
    if (phase == 1) { /* drain */
      /* concat_adr=0        adr0,L=0        | adr1,L=0        | adr2,L=0        */
      /* concat_adr=adr0,L=0 adr0,L=0,mark=0 | adr1,L=0        | adr2,L=0        */
      /* concat_adr=adr0,L=1          mark=0 | adr1,L=0,mark=0 | adr2,L=0        */
      /* concat_adr=adr0,L=2          mark=0 |          mark=0 | adr2,L=0,mark=1 */
//printf("drain: adr=%08.8x len=%08.8x nxt=%08.8x\n", (Uint)dmadr, (Uint)dmlen, (Uint)dmnxt);
      if ((emax7[LANE].lmmd[(m+1)%EMAX_DEPTH][j]&(1<<c)) && (dmadr+(dmlen+1)*sizeof(Uint)) == dmnxt) {
	if (!concat_adr[LANE][c]) { concat_adr[LANE][c] = dmadr; concat_len[LANE][c] = dmlen; }
	else                      { concat_len[LANE][c] += dmlen+1; }
	if (concat_len[LANE][c] < 8192) mark = 0;
      }
      else {
	if (concat_adr[LANE][c])  { concat_len[LANE][c] += dmlen+1; }
      }
    }
    else if (phase == 2) { /* load */
//printf("load: adr=%08.8x len=%08.8x nxt=%08.8x\n", (Uint)dmadr, (Uint)dmlen, (Uint)dmnxt);
      if (lmmicp1->v && (dmadr+(dmlen+1)*sizeof(Uint)) == dmnxt) {
	if (!concat_adr[LANE][c]) { concat_adr[LANE][c] = dmadr; concat_len[LANE][c] = dmlen; }
	else                      { concat_len[LANE][c] += dmlen+1; }
	if (concat_len[LANE][c] < 8192) mark = 0;
      }
      else {
	if (concat_adr[LANE][c])  { concat_len[LANE][c] += dmlen+1; }
      }
    }
#endif
  }

  /* dma */
  if (mark) {
    emax7[LANE].rw = dmrw;
    if (phase == 1) { /* drain */
      emax7[LANE].ddraddr = (concat_adr[LANE][c])?concat_adr[LANE][c]:dmadr; /* address should be 4B-aligned */
      emax7[LANE].lmmaddr = emax7[LANE].ddraddr;
      emax7[LANE].dmalen  = (concat_adr[LANE][c])?concat_len[LANE][c]:dmlen; /* length should be # of words */
    }
    else if (phase == 3 && dmrw==1) { /* pdrain */
      emax7[LANE].ddraddr = dmadr+(Sll)(int)lmmicp->ofs; /* ������PDRAIN address should be 4B-aligned */
      emax7[LANE].lmmaddr = emax7[LANE].ddraddr;
      emax7[LANE].dmalen  = dmlen; /* length should be # of words */
    }
    else if (phase == 2                /* load */
	  ||(phase == 3 && dmrw==0)) { /* pload *//* address should be 4B-aligned *//* length should be # of words */
      if (lmmicp->blk==0) { /* inf */
	if (phase == 2) { /* load */
	  emax7[LANE].ddraddr = (concat_adr[LANE][c])?concat_adr[LANE][c]:dmadr; /* address should be 4B-aligned */
	  emax7[LANE].lmmaddr = emax7[LANE].ddraddr;
	  emax7[LANE].dmalen  = (concat_adr[LANE][c])?concat_len[LANE][c]:dmlen; /* length should be # of words */
	}
	else {
	  emax7[LANE].ddraddr = dmadr+(Sll)(int)lmmicp->ofs; /* ������PLOAD address should be 4B-aligned */
	  emax7[LANE].lmmaddr = emax7[LANE].ddraddr;
	  emax7[LANE].dmalen  = dmlen; /* length should be # of words */
	}
#ifndef IGNORE_LMMI_BLKGATHER
	emax7[LANE].blksize    = 0; /* max:10bit */
#endif
      }
#ifndef IGNORE_LMMI_BLKGATHER
      else { /* 16,32,64 */
	if (phase == 2) /* load */
	  emax7[LANE].plist = dmadr+emax7[LANE].blkcount*8; /* address should be 4B-aligned */
	else
	  emax7[LANE].plist = dmadr+emax7[LANE].blkcount*8+(Sll)(int)lmmicp->ofs; /* ������PLOAD address should be 4B-aligned */
	emax7[LANE].blksize  = 32<<lmmicp->blk; /* max:10bit */
	if (emax7[LANE].blkcount==0) {
	  emax7[LANE].lmmblktop = 0; /* ������̤���������� ��Ƭ���ɥ쥹��0�ʤΤ�,addr_range�˹��פ�ɬ�� */
	  emax7[LANE].lmmblklen = dmlen; /* length should be # of words */
	}
	emax7[LANE].ddraddr    = emax7[LANE].plist; /* address should be 4B-aligned */
	emax7[LANE].lmmaddr    = emax7[LANE].lmmblktop;
	emax7[LANE].dmalen     = (emax7[LANE].lmmblklen<emax7[LANE].blksize)?emax7[LANE].lmmblklen:emax7[LANE].blksize-1;
	emax7[LANE].lmmblktop += emax7[LANE].blksize*sizeof(Ull);
	emax7[LANE].lmmblklen = (emax7[LANE].lmmblklen<emax7[LANE].blksize)?0:(emax7[LANE].lmmblklen-emax7[LANE].blksize);
	if (emax7[LANE].lmmblklen==0)
	  emax7[LANE].blkcount = 0;
	else
	  emax7[LANE].blkcount++; /* ������̤���������� continue ʣ�����DMA��ư��ɬ�� */
      }
#endif
    }
#if 0
printf("====LANE=%d DMA mode=%x phase=%x i=%x m=%x j=%x lmmic/o=%x/%x lmmc_stat=%x(dirty=%x) lmmo_stat=%x(dirty=%x) mark=%x", LANE, mode, phase, i, m, j, emax7[LANE].lmmic, emax7[LANE].lmmio, lmmc_stat, emax7[LANE].lmmd[i][j], lmmo_stat, emax7[LANE].lmmd[m][j], mark);
printf(" rw=0x%x ddraddr=%08.8x lmmaddr=%08.8x dmalen=0x%x\n", emax7[LANE].rw, (Uint)emax7[LANE].ddraddr, (Uint)emax7[LANE].lmmaddr, (Uint)emax7[LANE].dmalen);
#endif
    concat_adr[LANE][c] = 0;
    //pthread_mutex_lock(&axi_dma_mutex);
    emax7_kick_dma(LANE, j);
    //pthread_mutex_unlock(&axi_dma_mutex);
  }
}

emax7_sigwait(int LANE)
{
  /* If activated for DMA/EXEC, the speed is too slow due to many syscall */
  /* Only enq/deq should be managed by sigwait(). 20231218 Nakashima */
  int signo;
  if (emax7[LANE].sigwait) {
    *emax7[LANE].sigstat = 2; /* wait */
    sigwait(emax7[LANE].sigset, &signo);
    *emax7[LANE].sigstat = 1; /* run */
  }
}

emax7_kick_dma(int LANE, int j) /* col */
{
  int status_mm2s, status_s2mm;
  Ull dst, src;
  Uint pio_words, pio_loop, pio_i, pio_b4, pio_b8, pio_b16, pio_e4, pio_e8, pio_e16;

  if (!emax7[LANE].ddraddr)
    return (0);

  if (j != emax7[LANE].csel_save) {
    ((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].csel = j; /* DMA/LDDMQ/TRANS�Ѥ��о�col���å� */
    emax7[LANE].csel_save = j;
  }
#if defined(FPDDMA)
#define FPDDMA_DEFINED 1
#else
#define FPDDMA_DEFINED 0
#endif
  if (FPDDMA_DEFINED && emax7[LANE].dmalen > 1) { /* 4B/8B: ->PIO */
    /* kick dma_ctrl (Simple Mode) */
    if (emax7[LANE].rw == 0) { /* mem->lmm */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMACR = 0x00010001;
      *(Ull*)&(((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_SA) = emax7[LANE].ddraddr-emax_info[LANE].ddr_mmap+emax_info[LANE].ddr_phys; /* address should be 4B-aligned */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_LENGTH = (emax7[LANE].dmalen+1)*sizeof(Uint);                                         /* length should be # of words */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DMACR = 0x00010001;
      *(Ull*)&(((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DA) = emax7[LANE].lmmaddr-emax_info[LANE].ddr_mmap+emax_info[LANE].lmm_phys; /* (emax7[LANE].awaddr & ~(sizeof(Ull)*UNIT_WIDTH-1)) */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_LENGTH = (emax7[LANE].dmalen+1)*sizeof(Uint);                                         /* length should be # of words */
      do {
	/*emax7_sigwait(LANE);*/
	status_mm2s = ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMASR;
	status_s2mm = ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DMASR;
	if ((status_mm2s & 0x71) || (status_s2mm & 0x71)) {
	  ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMACR = 0x00010004; /* reset */
	  printf("emax7_check_lmmi_and_dma(): mem->lmm status_mm2s=%08x, status_s2mm=%8.8x (malfunction)\n", status_mm2s, status_s2mm);
	  break;
	}
      } while (!(status_mm2s & 0x2) || !(status_s2mm & 0x2));
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMASR = 0x00001000; /* clear */
    }
    else { /* lmm->mem */
      while (((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].stat & 0xffff00f0); //LMRING_BUSY
      ((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].dmrp = (1LL<<63)|((emax7[LANE].dmalen+1)*sizeof(Uint)<<40)|(emax7[LANE].lmmaddr-emax_info[LANE].ddr_mmap+emax_info[LANE].lmm_phys);
      /*printf("dmrp=%08.8x_%08.8x\n", (Uint)((((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].dmrp)>>32), (Uint)(((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].dmrp));*/
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMACR = 0x00010001;
      *(Ull*)&(((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_SA) = emax7[LANE].lmmaddr-emax_info[LANE].ddr_mmap+emax_info[LANE].lmm_phys; /* (emax7[LANE].awaddr & ~(sizeof(Ull)*UNIT_WIDTH-1)) */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_LENGTH = (emax7[LANE].dmalen+1)*sizeof(Uint);                                         /* length should be # of words */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DMACR = 0x00010001;
      *(Ull*)&(((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DA) = emax7[LANE].ddraddr-emax_info[LANE].ddr_mmap+emax_info[LANE].ddr_phys; /* address should be 4B-aligned */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_LENGTH = (emax7[LANE].dmalen+1)*sizeof(Uint);                                         /* length should be # of words */
      do {
	/*emax7_sigwait(LANE);*/
	status_mm2s = ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMASR;
	status_s2mm = ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->S2MM_DMASR;
	if ((status_mm2s & 0x71) || (status_s2mm & 0x71)) {
	  ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMACR = 0x00010004; /* reset */
	  printf("emax7_check_lmmi_and_dma(): lmm->mem status_mm2s=%08x, status_s2mm=%8.8x (malfunction)\n", status_mm2s, status_s2mm);
	  break;
	}
      } while (!(status_mm2s & 0x2) || !(status_s2mm & 0x2));
      /* end of DMAREADBUF */
      ((struct dma_ctrl*)emax7[LANE].dma_ctrl)->MM2S_DMASR = 0x00001000; /* clear */
      ((struct reg_ctrl*)emax7[LANE].reg_ctrl)->i[0].dmrp  = (0LL<<63); /* off */
    }
  }
  else { /* ���굡�¹Ԥˤ��cache-fill��ư���Ƥ��ޤ�.�굡���Ը��DMA��ư��,cache-fill�⤵���Τ�mismatch�Ȥʤ�� */
    /*printf("emax_info[LANE].lmm_mmap-emax_info[LANE].ddr_mmap=%08.8x_%08.8x\n", (Uint)((emax_info[LANE].lmm_mmap-emax_info[LANE].ddr_mmap)>>32), (Uint)(emax_info[LANE].lmm_mmap-emax_info[LANE].ddr_mmap));*/
    if (emax7[LANE].rw == 0) { /* mem->lmm */
      dst = emax7[LANE].lmmaddr-emax_info[LANE].ddr_mmap+emax_info[LANE].lmm_mmap;
      src = emax7[LANE].ddraddr;
#if 0
      printf("emax7[LANE].lmmaddr:%08.8x_%08.8x <- emax7[LANE].ddraddr:%08.8x_%08.8x\n",
	     (Uint)((Ull)((Ull*)emax7[LANE].lmmaddr)>>32), (Uint)(Ull)((Ull*)emax7[LANE].lmmaddr),
	     (Uint)((Ull)((Ull*)emax7[LANE].ddraddr)>>32), (Uint)(Ull)((Ull*)emax7[LANE].ddraddr));
#endif
    }
    else { /* lmm->mem */
      dst = emax7[LANE].ddraddr;
      src = emax7[LANE].lmmaddr-emax_info[LANE].ddr_mmap+emax_info[LANE].lmm_mmap;
#if 0
      printf("emax7[LANE].lmmaddr:%08.8x_%08.8x -> emax7[LANE].ddraddr:%08.8x_%08.8x\n",
	     (Uint)((Ull)((Ull*)emax7[LANE].lmmaddr)>>32), (Uint)(Ull)((Ull*)emax7[LANE].lmmaddr),
	     (Uint)((Ull)((Ull*)emax7[LANE].ddraddr)>>32), (Uint)(Ull)((Ull*)emax7[LANE].ddraddr));
#endif
    }
    /* src��dst��32B���饤�󤵤�Ƥ���,��¦�Τ��󥢥饤��ˤʤ뤳�ȤϤʤ� */
    pio_words = emax7[LANE].dmalen+1;
    if (src & (sizeof(Ull)-1) & sizeof(Uint)) { /* 4B-access 0,4 */
      *(Uint*)dst = *(Uint*)src;
      src += sizeof(Uint);
      dst += sizeof(Uint);
      pio_words--;
    }
    if (pio_words >= 2) {
      if (src & (sizeof(Dll)-1) & sizeof(Ull)) { /* 8B-access 0,4 */
	*(Ull*)dst = *(Ull*)src;
	src += sizeof(Ull);
	dst += sizeof(Ull);
	pio_words-=2;
      }
    }
    if (pio_words >= 4) {
      if (pio_loop = pio_words/(sizeof(Dll)/sizeof(Uint))) {
	for(pio_i=0; pio_i<pio_loop; pio_i++)
	  *((Dll*)dst + pio_i) = *((Dll*)src + pio_i);
	pio_words -= pio_loop*(sizeof(Dll)/sizeof(Uint));
	src += pio_loop*sizeof(Dll);
	dst += pio_loop*sizeof(Dll);
      }
    }
    if (pio_words >= 2) { /* 8B-access */
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull);
      dst += sizeof(Ull);
      pio_words-=2;
    }
    if (pio_words >= 1) { /* 4B-access */
      *(Uint*)dst = *(Uint*)src;
      src += sizeof(Uint);
      dst += sizeof(Uint);
      pio_words--;
    }
  }

  return (0);
}

/*******************************************************************************/
/******************************** EMAX7-START **********************************/
/*******************************************************************************/

/* lmmwb=0: if lmm never be written back to DDR */
emax7_pre_with_keep_cache()
{
  /* (conf -> scon -> addr -> breg ) -> dma -> exec -> dma -> term */
#ifdef ARMSIML
  emax_pre_with_keep_cache(); /* start syscall EMAX7 */
#endif
#ifdef ARMZYNQ
  /* do nothing */
#endif
}

/* lmmwb=1: if lmm may   be written back to DDR */
emax7_pre_with_drain_cache()
{
  /* (conf -> scon -> addr -> breg ) -> dma -> exec -> dma -> term */
#ifdef ARMSIML
  emax_pre_with_drain_cache(); /* start syscall EMAX7 */
#endif
#ifdef ARMZYNQ
  /* do nothing */
#endif
}

#endif

#ifdef __cplusplus
}
#endif
#endif