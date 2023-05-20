import java.io.*;
import java.util.*;
import java.lang.*;
public class turtle
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("turtle.in"));
      Queue<Integer> qrow = new ArrayDeque<>();
      Queue<Integer> qcol = new ArrayDeque<>();
      Queue<Integer> qmoves = new ArrayDeque<>();
      int m = in.nextInt(), n = in.nextInt(), m1 = in.nextInt(), m2 = in.nextInt();
      int[][] a = new int[m][n];
      boolean[][] visited = new boolean[m][n];
      for(int i = 0; i < m; i++)
         for(int j = 0; j < n; j++)
         {
            a[i][j] = in.nextInt();
            if(a[i][j] == 3)
            {
               visited[i][j] = true;
               qrow.add(i);
               qcol.add(j);
               qmoves.add(0);
            }
         }
      in.close();
      
      int[] di = {m1, m1, -1 * m1, -1 * m1, m2, m2, -1 * m2, -1 * m2};
      int[] dj = {m2, -1 * m2, m2, -1 * m2, m1, -1 * m1, m1, -1 * m1};
      int currentr, currentc, currentm;
      while(!qrow.isEmpty())
      {
         currentr = qrow.peek();
         qrow.remove();
         currentc = qcol.peek();
         qcol.remove();
         currentm = qmoves.peek();
         qmoves.remove();
         if(a[currentr][currentc] == 4)
         {
            System.out.println(currentm);
            break;
         }
         for(int i = 0; i < 8; i++)
         {
            if(currentr + di[i] >= 0 && currentr + di[i] < m && currentc + dj[i] >= 0 && currentc + dj[i] < n)
            {
               if((a[currentr + di[i]][currentc + dj[i]] == 1 || a[currentr + di[i]][currentc + dj[i]] == 4) && !visited[currentr + di[i]][currentc + dj[i]])
               {
                  qrow.add(currentr + di[i]);
                  qcol.add(currentc + dj[i]);
                  qmoves.add(currentm + 1);
                  visited[currentr + di[i]][currentc + dj[i]] = true;
               }
            }
         }
      }
   }
}