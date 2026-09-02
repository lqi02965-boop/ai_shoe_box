import time
from machine import UART
from machine import FPIOA

#帧定义
#帧头
FRAME_HEAD=0xAA
FRAME_TAIL=0x55
#命令字节
#32-->230
CMD_START_ENROLL=0x16  # 开始人脸录入
CMD_START_RECOG=0x17    #开始人脸识别
CMD_DELETE_FACE=0x18      #删除人脸
#230--->32
CMD_ENROLL_RESULT=0x25  #录入结果
CMD_RECOG_RESULT=0x26   #人脸识别结果
CMD_DELETE_RESULT=0x27  #删除结果

def init_uart():
    """初始化uart2 接32的usart3"""
    fpioa=FPIOA()
    fpioa.set_function(11,FPIOA.UART2_TXD)  # TX → STM32 PB11
    fpioa.set_function(12,FPIOA.UART2_RXD)# RX ← STM32 PB10
    '''
    初始化k230的串口 参数(使用uart2,波特率
                        数据位是8位 一个字节8bit每次传得就是一个完整字节
                        奇偶校验位 无(因为我们的自定义帧中是crc校验)
                        停止位1位 串口每传一个字节的结构是：起始位(1) + 数据位(8) + 停止位(1)，
                        把配好的uart对象返回出去
    '''

    uart=UART(UART.UART2,baudrate=115200,
              bits=UART.EIGHTBITS,
              parity=UART.PARITY_NONE,
              stop=UART.STOPBITS_ONE)
    return uart

def calc_crc8(data):
    """ crc8校验 多项式 0x07"""
    crc=0
    for b in data:
        crc ^=b
        for _ in range(8):
            if crc &0x80:
                crc=((crc<<1)^0x07)&0xFF
            else:
                crc=(crc<<1)&0xFF
    return crc

def send_frame(uart,cmd,data=b''):
    '''这个函数负责将数据打包发出去
            参数 uart(串口对象),cmd(命令字,比如0x26),data(数据域,默认空)
            长度=1字节命令+数据域字节
            把长度和命令字节和数据域拼在一起
            对payload算crc8得到亿字节校验码
            拼出完整的一帧
            将拼出的帧发出去(写入寄存器之后,只要寄存器中有字,自动发)
    '''
    data_len=1+len(data)
    payload=bytes([data_len,cmd])+data
    crc=calc_crc8(payload)
    frame=bytes([FRAME_HEAD])+payload+bytes([crc,FRAME_TAIL])
    uart.write(frame)

def read_frame(uart,timeout_ms=50):
    '''负责接收stm32发来的帧并解xi
    '''
    #接收缓冲区,把收到的字节收到这里
    buf=b''
    #记下开始时间戳
    start=time.ticks_ms()
    #50ms超时检测 超时了还没收到完整帧就放弃,返回None
    #ticks_diff(现在,开始)用来计算时间差
    while time.ticks_diff(time.ticks_ms(),start)<timeout_ms:
        #检查串口有没有数据进来
        if uart.any():
            #把收到的字节收到buf中,因为一次可能只收到几个字节所以等待
            buf  +=uart.read()
            #最小是五字节一帧 ,不够5字节证明没传完等待
            if len(buf)<5:

                continue
            #在buf中找帧头0xAA,因为可能会收到垃圾字节,所以从帧头开始算有效数据
            idx=buf.find(b'\xaa')
            #如果idx<0证明没有数据所以继续等待数据
            if idx<0:
                buf=b''
                continue

            #砍掉帧头前的垃圾,从帧头开始保留
            buf=buf[idx:]
            #如果buf长度小于5 数据没传完,继续等待
            if len(buf)<5:
                continue


            #读第二个字节拿到长度是几知道命令字节和数据域长度
            data_len=buf[1]
            #total就是这个帧的总长
            total=4+data_len
            #如果buf的长度小于帧的长度,表明没传完要继续传
            if len(buf)<total:
                continue
            #检查帧尾是不是0x55,不是则砍掉第一个字节继续找下一个帧头

            if buf[total -1]!=FRAME_TAIL:
                #这里就是帧尾不对就把帧头砍了,向后读知道找到下一个帧头
                buf=buf[1:]
                continue
            #如果上述全部正确进入下一步,也就是拿到cmd也就是命令字节
            cmd=buf[2]
            #所以就是 buf[3:6]，取下标3、4、5 三个字节：
            data=buf[3:2+data_len]
            #crc检测位,拿到数据
            crc_recv=buf[2+data_len]
            #计算收到的数据的crc
            crc_calc=calc_crc8(buf[1:2+data_len])
            #如果两个crc数据不对那么重新等待
            if crc_recv != crc_calc:
                buf=buf[1:]
                continue
            return (cmd,data)
    return None









