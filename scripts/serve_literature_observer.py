"""Read-only literature DEV observer; no model, no fake verbal thoughts."""
import sys
import serve_matched_observer as observer
if '--port' not in sys.argv:sys.argv.extend(['--port','8772'])
if sys.argv[sys.argv.index('--port')+1]=='8767':raise ValueError('Reserved port')
observer.PAGE=observer.PAGE.replace('Kiểm chứng R8','Literature-first · đối chiếu thật')
if __name__=='__main__':observer.main()
