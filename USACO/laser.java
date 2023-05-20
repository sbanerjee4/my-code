import java.io.*;
import java.util.*;
import java.lang.*;
public class laser
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("laser.in"));
      int n = in.nextInt();
      int m = in.nextInt();
      int[][] a = new int[m][n];
      int cRow = 0, cCol = 0;
      for(int i = 0; i < m; i++)
      {
         String temp = in.next();
         for(int j = 0; j < temp.length(); j++)
            if(temp.charAt(j) == 'C')
            {
               a[i][j] = -3;
               cRow = i;
               cCol = j;
            }
            else if(temp.charAt(j) == '.') a[i][j] = -1;
            else a[i][j] = -2;
      }
      in.close();
      
      int[] di = {0, 0, 1, -1};
      int[] dj = {-1, 1, 0, 0};
      Queue<Integer> qr = new ArrayDeque<>();
      Queue<Integer> qc = new ArrayDeque<>();
      a[cRow][cCol] = 0;
      for(int i = 0; i < 4; i++)
      {
         int r = cRow + di[i], c = cCol + dj[i];
         while(r >= 0 && r < m && c >= 0 && c < n && a[r][c] != -2)
         {
            a[r][c] = 0;
            qr.add(r);
            qc.add(c);
            r += di[i];
            c += dj[i];
         }
      }
      
      while(!qr.isEmpty())
      {
         int cr = qr.remove();
         int cc = qc.remove();
         int cm = a[cr][cc] + 1;
         for(int i = 0; i < 4; i++)
         {
            int r = cr + di[i], c = cc + dj[i];
            while(r >= 0 && r < m && c >= 0 && c < n && (a[r][c] == -1 || a[r][c] == -3 || a[r][c] == cm))
            {
               if(a[r][c] == -3)
               {
                  System.out.println(cm);
                  System.exit(0);
               }
               a[r][c] = cm;
               qr.add(r);
               qc.add(c);
               r += di[i];
               c += dj[i];
            }
         }
      }
   }
}