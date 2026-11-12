"""Separate passive observer. Never claims neuron activity is verbal thought."""
import sys
import serve_matched_observer as observer
if '--port' not in sys.argv:sys.argv.extend(['--port','8771'])
if sys.argv[sys.argv.index('--port')+1]=='8767':raise ValueError('Port8767 is reserved; do not touch')
observer.PAGE=observer.PAGE.replace('Kiểm chứng R8','Adaptive Credit Routing · DEV thật')
if __name__=='__main__':observer.main()
